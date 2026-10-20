"""Span helpers for the chat agent loop (turn → iteration → LLM / tools).

Resulting tree for one ``/v1/messages`` turn::

    POST /v1/messages                    (FastAPI server span)
    └─ chat.turn                [AGENT]  session.id, user.id, execution id
       ├─ chat.iteration 1      [CHAIN]  tools offered, stop reason, tokens
       │  ├─ <LLM>.astream_chat [LLM]    (OpenInference llama-index)
       │  └─ tool <name>        [TOOL]   arguments, result, status
       └─ chat.iteration 2 ...
"""

from __future__ import annotations

from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any

from private_gpt.observability.message import (
    fold_user_message,
    last_user_message,
    message_text,
)
from private_gpt.observability.processors import (
    register_llm_tools,
    unregister_llm_tools,
)
from private_gpt.observability.tracing import (
    AGENT_NAME,
    GRAPH_NODE_ID,
    GRAPH_NODE_NAME,
    GRAPH_NODE_PARENT_ID,
    LLM_MODEL_NAME,
    TOKEN_COMPLETION,
    TOKEN_PROMPT,
    TOKEN_TOTAL,
    TOOL_CALL_ID,
    TOOL_DESCRIPTION,
    TOOL_NAME,
    TOOL_PARAMETERS,
    SpanKind,
    is_enabled,
    set_span_attributes,
    start_span,
    using_trace_attributes,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from private_gpt.observability.message import MessageFolder

AGENT = "chat_agent"


def user_input_attributes(messages: Any) -> dict[str, Any]:
    """Folded last user message as span input (same on every chat span).

    ``input.value`` is the readable text (``[image: image/png]`` markers for
    media), which session views show as the user bubble; the OpenInference
    ``llm.input_messages`` contents let UIs render images; the full block list
    is in ``pgpt.input.message``.
    """
    message = last_user_message(messages or [])
    if message is None:
        return {}
    attributes: dict[str, Any] = {
        "input": message_text(message),
        "pgpt.input.message": message,
        "llm.input_messages.0.message.role": "user",
    }
    prefix = "llm.input_messages.0.message.contents"
    for index, block in enumerate(message["content"]):
        if block["type"] == "text":
            attributes[f"{prefix}.{index}.message_content.type"] = "text"
            attributes[f"{prefix}.{index}.message_content.text"] = block["text"]
        elif block["type"] == "image" and block.get("url"):
            attributes[f"{prefix}.{index}.message_content.type"] = "image"
            attributes[f"{prefix}.{index}.message_content.image.image.url"] = block[
                "url"
            ]
    return attributes


@contextmanager
def chat_turn_span(
    name: str,
    request: Any,
    execution_id: str,
    *,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[Any]:
    """AGENT span for one chat turn plus session/user context for children."""
    context = getattr(request, "context", None)
    system = getattr(request, "system", None)
    # A session is a conversation: the caller's container (the backend sends
    # the thread id). Without one each turn stands alone. The user id is only
    # what the caller sent in metadata.user_id; the request mapper fills a
    # random uuid when it is missing, which must not become a "user".
    session_id = getattr(context, "container", None) or execution_id
    user_id = (
        getattr(context, "user_id", None)
        if getattr(context, "user_id_provided", False)
        else None
    )
    with (
        using_trace_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata={"execution_id": execution_id},
        ),
        start_span(
            name,
            SpanKind.AGENT,
            attributes={
                AGENT_NAME: AGENT,
                GRAPH_NODE_ID: AGENT,
                GRAPH_NODE_NAME: AGENT,
                LLM_MODEL_NAME: getattr(system, "model", None),
                "pgpt.execution_id": execution_id,
                "pgpt.message_count": len(getattr(request, "messages", []) or []),
                "pgpt.thinking.enabled": getattr(
                    getattr(request, "thinking", None), "enabled", None
                ),
                **(attributes or {}),
            },
        ) as span,
    ):
        set_span_attributes(
            span, **user_input_attributes(getattr(request, "messages", None))
        )
        try:
            yield span
        except Exception as exc:
            # Session views show the output as the answer bubble; a failed
            # turn shows why instead of an empty one.
            set_span_attributes(span, output=f"Error: {exc}")
            raise


def record_turn_result(
    span: Any, state: Any, folder: MessageFolder | None = None
) -> None:
    """Final result on the turn span.

    ``output.value`` is the answer text the user read: backend session views
    render it as the assistant bubble. The whole folded
    message (text, reasoning, tool calls and results, stop reason, usage) is
    in ``pgpt.output.message`` as JSON for inspection.
    """
    output = getattr(state, "output", None)
    runtime = getattr(state, "runtime", None)
    message = folder.message() if folder is not None else None
    if message is not None:
        message["stop_reason"] = message.get("stop_reason") or getattr(
            output, "stop_reason", None
        )
    text = folder.text if folder is not None else ""
    reasoning = folder.reasoning if folder is not None else ""
    set_span_attributes(
        span,
        output=text or None,
        **{
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": text or None,
            "pgpt.output.message": message,
            "pgpt.output.reasoning": reasoning or None,
            "pgpt.output.tool_calls": [
                block.get("name")
                for block in (message or {}).get("content", [])
                if block.get("type") in ("tool_use", "server_tool_use")
            ]
            or None,
            "pgpt.status": str(getattr(output, "status", "")),
            "pgpt.stop_reason": getattr(output, "stop_reason", None),
            "pgpt.iterations": getattr(runtime, "iteration", None),
            "pgpt.pending_async_tools": len(
                getattr(output, "pending_async_tools", {}) or {}
            ),
        },
    )
    _set_tokens(
        span,
        getattr(runtime, "total_input_tokens", 0) if runtime else 0,
        getattr(runtime, "total_output_tokens", 0) if runtime else 0,
    )


def _set_tokens(span: Any, prompt: int | None, completion: int | None) -> None:
    if not prompt and not completion:
        return
    set_span_attributes(
        span,
        **{
            TOKEN_PROMPT: prompt or 0,
            TOKEN_COMPLETION: completion or 0,
            TOKEN_TOTAL: (prompt or 0) + (completion or 0),
        },
    )


@contextmanager
def iteration_span(
    iteration: int, tools: list[Any], messages: Any = None
) -> Iterator[Any]:
    """CHAIN span around one LLM call and the tools it triggers."""
    with start_span(
        f"chat.iteration {iteration}",
        SpanKind.CHAIN,
        attributes={
            "pgpt.iteration": iteration,
            "pgpt.tools.offered": [
                getattr(getattr(t, "metadata", None), "name", "") for t in tools
            ],
            GRAPH_NODE_ID: f"iteration_{iteration}",
            GRAPH_NODE_NAME: f"iteration {iteration}",
            GRAPH_NODE_PARENT_ID: AGENT,
        },
    ) as span:
        set_span_attributes(span, **user_input_attributes(messages))
        if is_enabled() and tools:
            # Real schemas for the child LLM span (the OpenInference handler
            # exports `<FunctionTool object at 0x..>` otherwise).
            schemas = []
            for tool in tools:
                try:
                    schemas.append(tool.metadata.to_openai_tool(skip_length_check=True))
                except Exception:
                    schemas.append({"name": getattr(tool.metadata, "name", "")})
            register_llm_tools(span, schemas)
        try:
            yield span
        finally:
            if is_enabled():
                unregister_llm_tools(span)


def record_iteration_state(span: Any, state: Any) -> None:
    """The state an iteration starts from, as its span input."""
    if is_enabled():
        set_span_attributes(span, input=state_snapshot(state))


def record_iteration_result(
    span: Any,
    *,
    stop_reason: str | None,
    tool_calls: int,
    input_tokens: int | None,
    output_tokens: int | None,
    state: Any = None,
) -> None:
    """Outcome of an iteration; ``state`` (the state it leaves) is the output.

    Input and output of consecutive iterations chain: what one leaves is what
    the next one starts from, so a trace replays the whole loop.
    """
    set_span_attributes(
        span,
        **{"pgpt.stop_reason": stop_reason, "pgpt.tool_calls": tool_calls},
    )
    if state is not None and is_enabled():
        set_span_attributes(span, output=state_snapshot(state))
    _set_tokens(span, input_tokens, output_tokens)


def state_snapshot(state: Any) -> dict[str, Any]:
    """Readable view of a chat loop state: conversation, tools, parameters.

    Messages keep text, reasoning, tool calls and tool results; media are
    summarized (small images inline). Runtime caches, tokenizers and the
    context stack internals are left out.
    """
    input_state = getattr(state, "input", None)
    request = getattr(input_state, "request", None)
    runtime = getattr(state, "runtime", None)
    output = getattr(state, "output", None)
    snapshot: dict[str, Any] = {
        "iteration": getattr(runtime, "iteration", None),
        "status": str(getattr(output, "status", "") or "") or None,
        "stop_reason": getattr(output, "stop_reason", None),
    }
    if request is not None:
        system = getattr(request, "system", None)
        prompt = getattr(system, "prompt", None)
        if isinstance(prompt, list):
            prompt = "\n\n".join(getattr(block, "text", "") for block in prompt)
        snapshot["system"] = prompt
        snapshot["messages"] = [
            state_message(message) for message in getattr(request, "messages", [])
        ]
        thinking = getattr(request, "thinking", None)
        if thinking is not None and getattr(thinking, "enabled", False):
            snapshot["thinking"] = getattr(thinking, "type", None)
    stack = getattr(input_state, "context_stack", None)
    if stack is not None and hasattr(stack, "all_tools"):
        with suppress(Exception):
            snapshot["tools"] = [
                {
                    key: value
                    for key, value in (
                        ("name", tool.name),
                        ("type", tool.type),
                        ("runtime", tool.runtime),
                    )
                    if value
                }
                for tool in stack.all_tools()
            ]
    params = getattr(input_state, "llm_kwargs", None)
    if params is not None:
        with suppress(Exception):
            snapshot["llm_params"] = params.model_dump(mode="json", exclude_none=True)
    pending = getattr(output, "pending_external_tool_calls", None)
    if pending:
        snapshot["pending_tool_calls"] = [_tool_call(call) for call in pending]
    if pending_async := getattr(output, "pending_async_tools", None):
        snapshot["pending_async_tools"] = dict(pending_async)
    if runtime is not None:
        snapshot["tokens"] = {
            "input": getattr(runtime, "total_input_tokens", 0),
            "output": getattr(runtime, "total_output_tokens", 0),
        }
    return {key: value for key, value in snapshot.items() if value is not None}


def state_message(message: Any) -> dict[str, Any]:
    """One llama-index message as role, content, reasoning, tool calls/results."""
    role = getattr(message, "role", None)
    role = str(getattr(role, "value", role))
    kwargs = getattr(message, "additional_kwargs", None) or {}
    folded = fold_user_message(message) if role == "user" else None
    item: dict[str, Any] = {"role": role}
    if folded is not None and any(b["type"] != "text" for b in folded["content"]):
        item["content"] = folded["content"]
    else:
        item["content"] = getattr(message, "content", None) or ""
    thinking = kwargs.get("thinking")
    if isinstance(thinking, str) and thinking.strip():
        item["reasoning"] = thinking
    calls = kwargs.get("tool_calls")
    if calls:
        item["tool_calls"] = [_tool_call(call) for call in calls]
    if role == "tool":
        item["tool_call_id"] = kwargs.get("tool_call_id")
        item["name"] = kwargs.get("tool_call_name")
        if kwargs.get("is_error"):
            item["is_error"] = True
    return {key: value for key, value in item.items() if value is not None}


def _tool_call(call: Any) -> Any:
    if isinstance(call, dict):
        return call
    return {
        "id": getattr(call, "tool_id", None),
        "name": getattr(call, "tool_name", None),
        "arguments": getattr(call, "tool_kwargs", None),
    }


@contextmanager
def tool_span(
    *,
    tool_name: str,
    tool_id: str,
    arguments: Any,
    description: str | None = None,
    parameters: Any = None,
    runtime: str | None = None,
) -> Iterator[Any]:
    with start_span(
        f"tool {tool_name}",
        SpanKind.TOOL,
        input=arguments,
        attributes={
            TOOL_NAME: tool_name,
            TOOL_CALL_ID: tool_id,
            TOOL_DESCRIPTION: description,
            TOOL_PARAMETERS: parameters,
            "pgpt.tool.runtime": runtime,
            GRAPH_NODE_ID: f"tool_{tool_name}",
            GRAPH_NODE_NAME: tool_name,
        },
    ) as span:
        yield span


@contextmanager
def interceptor_span(interceptor: Any, phase: Any) -> Iterator[Any]:
    """CHAIN span for one interceptor run (condensation, documents, media...).

    Spans of interceptors that changed nothing are dropped on export (see
    :class:`~private_gpt.observability.processors.DropSpansProcessor`), so
    each iteration only shows the steps that did work.
    """
    name = type(interceptor).__name__
    with start_span(
        f"interceptor {name}",
        SpanKind.CHAIN,
        attributes={
            "pgpt.interceptor": name,
            "pgpt.interceptor.phase": str(getattr(phase, "value", phase)),
        },
    ) as span:
        yield span


def state_fingerprint(state: Any) -> tuple[Any, ...] | None:
    """Cheap identity snapshot of what an interceptor can change.

    Interceptors mutate the state in place, so comparing the state object
    before/after says nothing; this records, before the call, the message
    objects with their content and kwargs keys, the system prompt and the
    context stack size, to compare with the state after it.
    """
    if not is_enabled():
        return None
    request = getattr(getattr(state, "input", None), "request", None)
    messages = tuple(
        (
            id(m),
            m.content if isinstance(m.content, str) else None,
            tuple(m.additional_kwargs),
        )
        for m in _messages(state)
    )
    system = getattr(getattr(request, "system", None), "prompt", None)
    return (messages, _stack_size(state), str(system) if system is not None else None)


def record_interceptor_result(span: Any, before: Any, after: Any) -> None:
    """Record what an interceptor changed: messages, system prompt, context."""
    if not is_enabled() or before is None:
        return
    after_fp = state_fingerprint(after)
    before_messages, before_stack, before_system = before
    after_messages, after_stack, after_system = after_fp or ((), 0, None)
    changes = [
        name
        for name, changed in (
            ("messages", before_messages != after_messages),
            ("system_prompt", before_system != after_system),
            ("context", before_stack != after_stack),
        )
        if changed
    ]
    set_span_attributes(
        span,
        **{
            "pgpt.interceptor.changed": bool(changes),
            "pgpt.interceptor.changes": changes or None,
            "pgpt.messages.before": len(before_messages),
            "pgpt.messages.after": len(after_messages),
        },
    )


def _messages(state: Any) -> list[Any]:
    request = getattr(getattr(state, "input", None), "request", None)
    return list(getattr(request, "messages", None) or [])


def _stack_size(state: Any) -> int:
    stack = getattr(getattr(state, "input", None), "context_stack", None)
    return len(getattr(stack, "layers", None) or [])
