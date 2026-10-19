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

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

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


def _last_user_text(request: Any) -> str | None:
    """Text of the last user message that has any (tool results have none)."""
    for message in reversed(getattr(request, "messages", []) or []):
        role = getattr(message, "role", None)
        if str(getattr(role, "value", role)) != "user":
            continue
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        texts = [
            text
            for block in getattr(message, "blocks", None) or []
            if isinstance(text := getattr(block, "text", None), str) and text.strip()
        ]
        if texts:
            return "\n".join(texts)
    return None


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
            input=_last_user_text(request),
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
        yield span


def record_turn_result(
    span: Any, state: Any, folder: MessageFolder | None = None
) -> None:
    """Final result on the turn span.

    ``output.value`` is the answer text the user read: session views (Phoenix
    sessions, Langfuse) render it as the assistant bubble. The whole folded
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
def iteration_span(iteration: int, tools: list[Any]) -> Iterator[Any]:
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


def record_iteration_result(
    span: Any,
    *,
    stop_reason: str | None,
    tool_calls: int,
    input_tokens: int | None,
    output_tokens: int | None,
) -> None:
    set_span_attributes(
        span,
        **{"pgpt.stop_reason": stop_reason, "pgpt.tool_calls": tool_calls},
    )
    _set_tokens(span, input_tokens, output_tokens)


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
