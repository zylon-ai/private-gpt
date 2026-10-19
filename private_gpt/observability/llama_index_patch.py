"""Keep the OpenInference llama-index handler cheap on the event loop.

The handler builds span attributes synchronously when a llama-index span
starts, i.e. on the request's event loop, before the LLM call. Profiling a
chat with a long history and an image showed three hot spots (~20 ms per LLM
call):

* ``process_input`` JSON-dumps every bound argument, including the whole
  message list, into ``input.value``. LLM spans already get the structured
  ``llm.input_messages``, so for LLMs it is skipped.
* ``_detect_llm_provider`` tries several provider imports on every span; it
  is cached per class.
* Nodes (documents, chunks, retrieval results) were rendered as
  ``<ChunkNode(id_=...)>``; they now show id, type, a text preview, score and
  the identifying metadata (artifact, file, page) as JSON.
* A streaming LLM that consumes another LLM stream (Triton ``astream_chat``
  wraps ``astream_complete``) was ended when the inner stream finished,
  before its own end event set the output; LLM spans now wait for their own
  end event.
* Tool calls are ``ToolSelection`` objects, which the handler skips: an
  assistant turn that only called tools showed as an empty message. They
  are written as OpenInference ``message.tool_calls`` (name, id, arguments),
  tool results get their ``tool_call_id`` and name, and reasoning goes to
  the message's ``thinking`` content.
* ``_process_messages`` writes every message of the prompt. Only the system
  messages and the last ``max_input_messages`` are kept; earlier turns are
  on the previous traces of the session, and the count of omitted ones is
  recorded.

Each patch checks the handler still has the expected shape and is skipped
otherwise, so an upgrade of the instrumentation degrades to its defaults.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_patched = False


def patch_llama_index_handler(max_input_messages: int) -> None:
    global _patched
    if _patched:
        return
    try:
        from llama_index.core.base.llms.base import BaseLLM
        from openinference.instrumentation.llama_index import _handler
    except ImportError:
        return

    span_cls = getattr(_handler, "_Span", None)
    if span_cls is None:
        logger.warning("llama-index handler layout changed; perf patches skipped")
        return

    original_input = getattr(span_cls, "process_input", None)
    if original_input is not None:

        def process_input(self: Any, instance: Any, bound_args: Any) -> None:
            if isinstance(instance, BaseLLM):
                return
            original_input(self, instance, bound_args)

        span_cls.process_input = process_input

    detect = getattr(_handler, "_detect_llm_provider", None)
    if detect is not None:
        providers: dict[type, str | None] = {}

        def detect_llm_provider(instance: Any) -> str | None:
            cls = type(instance)
            if cls not in providers:
                providers[cls] = detect(instance)
            return providers[cls]

        setattr(_handler, "_detect_llm_provider", detect_llm_provider)  # noqa: B010

    show_repr = getattr(_handler, "_show_repr_str", None)
    if show_repr is not None:

        def show_repr_str(obj: Any) -> Any:
            summary = node_summary(obj)
            return summary if summary is not None else show_repr(obj)

        setattr(_handler, "_show_repr_str", show_repr_str)  # noqa: B010

    encoder = getattr(_handler, "_encoder", None)
    if encoder is not None:
        # Nodes nested in lists/dicts (retriever results, transform inputs)
        # are encoded as objects, not as JSON strings inside JSON.
        def encode(obj: Any) -> Any:
            info = node_info(obj)
            return info if info is not None else encoder(obj)

        setattr(_handler, "_encoder", encode)  # noqa: B010

    notify_parent = getattr(span_cls, "notify_parent", None)
    finished = getattr(getattr(_handler, "_StreamingStatus", None), "FINISHED", None)
    llm_kind = getattr(_handler, "LLM", None)
    if notify_parent is not None and finished is not None and llm_kind is not None:

        def notify(self: Any, status: Any) -> None:
            parent = getattr(self, "_parent", None)
            if (
                status is finished
                and parent is not None
                and getattr(parent, "_span_kind", None) == llm_kind
                and parent.active
                and parent.waiting_for_streaming
            ):
                # The parent LLM span ends on its own LLM*EndEvent, which
                # carries the output messages.
                return
            notify_parent(self, status)

        span_cls.notify_parent = notify  # ty: ignore[invalid-assignment]

    original_event = getattr(span_cls, "process_event", None)
    output_prefix = getattr(_handler, "LLM_OUTPUT_MESSAGES", None)
    if original_event is not None and output_prefix is not None:
        from llama_index.core.instrumentation.events.llm import (
            LLMChatEndEvent,
            LLMChatInProgressEvent,
            LLMChatStartEvent,
            LLMCompletionEndEvent,
            LLMCompletionInProgressEvent,
            LLMCompletionStartEvent,
        )

        from private_gpt.observability.tracing import pending_invocation_parameters

        def process_event(self: Any, event: Any) -> None:
            # Parameters the backend actually sent (set while the request is
            # built), replacing the static metadata the instrumentation shows.
            if isinstance(event, (LLMChatStartEvent, LLMCompletionStartEvent)):
                pending_invocation_parameters.set(None)
            elif isinstance(event, (LLMChatEndEvent, LLMCompletionEndEvent)):
                if parameters := pending_invocation_parameters.get():
                    self["llm.invocation_parameters"] = parameters
            # Streamed tool calls and reasoning come in some chunks, not
            # necessarily the last one the end event carries; collect them as
            # they pass.
            if isinstance(
                event, (LLMChatInProgressEvent, LLMCompletionInProgressEvent)
            ):
                _collect_tool_calls(self, _tool_calls(event.response))
                _collect_reasoning(self, event.response)
            elif isinstance(event, LLMChatEndEvent) and event.response is not None:
                _collect_tool_calls(self, _tool_calls(event.response))
                extra = _streamed_extras(self, event.response)
                if extra:
                    message = event.response.message
                    message = message.model_copy(
                        update={
                            "additional_kwargs": {
                                **message.additional_kwargs,
                                **extra,
                            }
                        }
                    )
                    event = event.model_copy(
                        update={
                            "response": event.response.model_copy(
                                update={"message": message}
                            )
                        }
                    )
            elif (
                isinstance(event, LLMCompletionEndEvent) and event.response is not None
            ):
                # Completions only record the text; show the answer as an
                # assistant message with its tool calls and reasoning, like
                # chat spans.
                _collect_tool_calls(self, _tool_calls(event.response))
                extra = _streamed_extras(self, event.response)
                if extra and output_prefix is not None:
                    from llama_index.core.base.llms.types import ChatMessage

                    self._process_messages(
                        output_prefix,
                        ChatMessage(
                            role="assistant",
                            content=event.response.text or "",
                            additional_kwargs=extra,
                        ),
                    )
            if isinstance(event, LLMChatEndEvent) and event.response is not None:
                # Chat responses converted from a completion stream keep the
                # usage on the message, where the handler does not look. Set
                # before the handler runs: it ends a streamed span on this
                # event, and usage it finds itself still takes precedence.
                message = getattr(event.response, "message", None)
                usage = getattr(message, "additional_kwargs", None) or {}
                for key, value in _token_counts(usage):
                    self[key] = value
            original_event(self, event)

        span_cls.process_event = process_event  # ty: ignore[invalid-assignment]

    original_end = getattr(span_cls, "end", None)
    if original_end is not None:

        def end(self: Any, exception: BaseException | None = None) -> None:
            # For a streamed LLM call the handler stamps the end time when the
            # generator is returned; the call really ends when it is drained.
            if getattr(self, "waiting_for_streaming", False):
                last = getattr(self, "_last_updated_at", None)
                # The sweeper ends abandoned streams after 60 s idle: use the
                # last chunk, not the sweep time.
                if last is not None and time.time() - last > 30:
                    self._end_time = max(self._end_time, int(last * 1e9))
                else:
                    self._end_time = time.time_ns()
            original_end(self, exception)

        span_cls.end = end  # ty: ignore[invalid-assignment]

    original_messages = getattr(span_cls, "_process_messages", None)
    input_prefix = getattr(_handler, "LLM_INPUT_MESSAGES", None)
    if original_messages is not None and input_prefix is not None:

        def process_messages(self: Any, prefix: str, *messages: Any) -> None:
            omitted = 0
            if prefix == input_prefix and len(messages) > max_input_messages:
                system = [m for m in messages if _role(m) == "system"]
                rest = [m for m in messages if _role(m) != "system"]
                kept = rest[-max_input_messages:]
                omitted = len(rest) - len(kept)
                messages = (*system, *kept)
            original_messages(self, prefix, *messages)
            for index, message in enumerate(messages):
                for key, value in _message_extras(message):
                    self[f"{prefix}.{index}.{key}"] = value
                    if key == "message.reasoning" and prefix == output_prefix:
                        # Trace UIs only render `content`; an answer that is
                        # all reasoning (before a tool call) would look empty.
                        content = getattr(message, "content", None) or ""
                        self[f"{prefix}.{index}.message.content"] = _quote_reasoning(
                            value, content
                        )
            if omitted:
                self["pgpt.llm.input_messages.omitted"] = omitted

        span_cls._process_messages = process_messages

    _patched = True


def _role(message: Any) -> str:
    role = getattr(message, "role", None)
    return str(getattr(role, "value", role))


# Metadata keys that identify where a node comes from; the rest (extracted
# terms, layout, relationships) is noise in a trace.
_NODE_METADATA = (
    "artifact",
    "collection",
    "file_name",
    "title",
    "page_label",
    "page",
    "sheet_name",
    "url",
)
_PREVIEW_CHARS = 300


def node_summary(obj: Any) -> str | None:
    """JSON summary of a llama-index node or ``NodeWithScore``; else None."""
    info = node_info(obj)
    if info is None:
        return None
    import json

    return json.dumps(info, ensure_ascii=False, default=str)


def node_info(obj: Any) -> dict[str, Any] | None:
    """Summary of a llama-index node or ``NodeWithScore``; else None."""
    try:
        from llama_index.core.schema import BaseNode, NodeWithScore
    except ImportError:
        return None
    score = None
    if isinstance(obj, NodeWithScore):
        score, obj = obj.score, obj.node
    if not isinstance(obj, BaseNode):
        return None
    summary: dict[str, Any] = {"type": type(obj).__name__, "id": obj.node_id}
    if score is not None:
        summary["score"] = round(float(score), 4)
    metadata = getattr(obj, "metadata", None) or {}
    picked = {
        k: metadata[k] for k in _NODE_METADATA if metadata.get(k) not in (None, "")
    }
    if picked:
        summary["metadata"] = picked
    text = getattr(obj, "text", None)
    if isinstance(text, str) and text.strip():
        summary["chars"] = len(text)
        summary["text"] = (
            text[:_PREVIEW_CHARS] + "…" if len(text) > _PREVIEW_CHARS else text
        )
    children = getattr(obj, "child_nodes", None) or getattr(obj, "children", None)
    if isinstance(children, list) and children:
        summary["children"] = len(children)
    return summary


def _message_extras(message: Any) -> list[tuple[str, Any]]:
    """Tool calls, tool-result ids and reasoning the handler leaves out."""
    import json

    extras: list[tuple[str, Any]] = []
    kwargs = getattr(message, "additional_kwargs", None) or {}
    for index, call in enumerate(kwargs.get("tool_calls") or []):
        name = getattr(call, "tool_name", None)
        if name is None:
            continue  # dicts and ToolCallBlocks are handled by the handler
        prefix = f"message.tool_calls.{index}.tool_call"
        extras.append((f"{prefix}.function.name", name))
        if tool_id := getattr(call, "tool_id", None):
            extras.append((f"{prefix}.id", tool_id))
        arguments = getattr(call, "tool_kwargs", None)
        extras.append(
            (
                f"{prefix}.function.arguments",
                arguments
                if isinstance(arguments, str)
                else json.dumps(arguments, ensure_ascii=False, default=str),
            )
        )
    if _role(message) == "tool":
        if name := kwargs.get("tool_call_name"):
            extras.append(("message.name", name))
    thinking = kwargs.get("thinking") or kwargs.get("reasoning_content")
    if isinstance(thinking, str) and thinking.strip():
        extras.append(("message.reasoning", thinking))
    return extras


def _collect_tool_calls(span: Any, calls: list[Any] | None) -> None:
    """Keep every tool call seen in the stream, once (by id, else position)."""
    if not calls:
        return
    collected = span.__dict__.setdefault("_pgpt_tool_calls", {})
    for index, call in enumerate(calls):
        key = (
            getattr(call, "tool_id", None)
            or (call.get("id") if isinstance(call, dict) else None)
            or f"#{index}"
        )
        collected[key] = call


def _quote_reasoning(reasoning: str, content: str) -> str:
    """Reasoning as a markdown quote, then a rule and the answer."""
    quoted = "\n".join(f"> {line}" if line else ">" for line in reasoning.splitlines())
    return quoted + (f"\n\n---\n\n{content}" if content else "")


def _token_counts(usage: dict[str, Any]) -> list[tuple[str, int]]:
    prompt = usage.get("input_tokens") or usage.get("prompt_tokens")
    completion = usage.get("output_tokens") or usage.get("completion_tokens")
    counts: list[tuple[str, int]] = []
    if isinstance(prompt, int):
        counts.append(("llm.token_count.prompt", prompt))
    if isinstance(completion, int):
        counts.append(("llm.token_count.completion", completion))
    if isinstance(prompt, int) and isinstance(completion, int):
        counts.append(("llm.token_count.total", prompt + completion))
    return counts


def _response_kwargs(response: Any) -> dict[str, Any]:
    """``additional_kwargs`` of a chat/completion response and its message."""
    message = getattr(response, "message", None)
    return {
        **(getattr(response, "additional_kwargs", None) or {}),
        **(getattr(message, "additional_kwargs", None) or {}),
    }


def _collect_reasoning(span: Any, response: Any) -> None:
    delta = _response_kwargs(response).get("thinking_delta")
    if isinstance(delta, str) and delta:
        span.__dict__.setdefault("_pgpt_reasoning", []).append(delta)


def _streamed_extras(span: Any, response: Any) -> dict[str, Any]:
    """Tool calls and reasoning seen in the stream, missing from the last chunk."""
    kwargs = _response_kwargs(response)
    extra: dict[str, Any] = {}
    if calls := span.__dict__.get("_pgpt_tool_calls"):
        extra["tool_calls"] = list(calls.values())
    thinking = kwargs.get("thinking") or kwargs.get("reasoning_content")
    if not (isinstance(thinking, str) and thinking.strip()):
        thinking = "".join(span.__dict__.get("_pgpt_reasoning", []))
    if thinking.strip():
        extra["thinking"] = thinking
    return extra


def _tool_calls(response: Any) -> list[Any] | None:
    kwargs = getattr(getattr(response, "message", None), "additional_kwargs", None)
    calls = (kwargs or {}).get("tool_calls") or (
        getattr(response, "additional_kwargs", None) or {}
    ).get("tool_calls")
    return list(calls) if calls else None
