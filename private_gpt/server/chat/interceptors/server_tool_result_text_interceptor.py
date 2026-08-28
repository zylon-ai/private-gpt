from __future__ import annotations

from typing import TYPE_CHECKING, Any

from injector import singleton
from llama_index.core.base.llms.types import MessageRole

from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.events.models import (
    NO_TOOL_CONTENT,
    TextBlock,
    normalize_tool_result_content,
    to_llama_index_blocks,
)
from private_gpt.events.models._tool_result_blocks import Renderable

if TYPE_CHECKING:
    from llama_index.core.base.llms.types import ChatMessage
    from llama_index.core.base.llms.types import ContentBlock as LIContentBlock

    from private_gpt.components.engines.chat.models.chat_interceptor_context import (
        ChatInterceptorContext,
    )

_EXCLUDED_KWARGS = frozenset(
    {
        "tool_call_id",
        "tool_call_name",
        "tool_call_args",
        "raw_output",
        "tldr",
    }
)


def _server_result_keys(kwargs: dict[str, Any]) -> list[str]:
    return [
        key
        for key, value in kwargs.items()
        if key not in _EXCLUDED_KWARGS
        and isinstance(value, list)
        and any(isinstance(item, Renderable) for item in value)
    ]


def _rendered_blocks(kwargs: dict[str, Any], keys: list[str]) -> list:
    rendered: list = []
    for key in keys:
        for block in kwargs.get(key, []):
            if isinstance(block, Renderable):
                rendered_text = block.render()
                if rendered_text.strip():
                    rendered.append(TextBlock(text=rendered_text))
            else:
                rendered.append(block)
    return rendered


@singleton
class ServerToolResultTextInterceptor(ChatRequestLoopInterceptor):
    """Project renderable tool results onto LLM-facing text without mutating history.

    Structured payloads live in ``additional_kwargs`` for every tool (bash results,
    skill JSON wrappers, ``server_tool_result``, …). Later interceptors reconstruct
    tool state from those kwargs, so this interceptor never deletes them and never
    mutates the original ``ChatMessage``. It only copies a message when the visible
    ``blocks`` need to be replaced with ``render()`` text.
    """

    async def intercept(self, context: ChatInterceptorContext) -> None:
        if context.phase != InterceptorPhase.BEFORE_ITERATION:
            return

        state = context.state
        messages = state.input.request.messages
        rewritten: list[ChatMessage] | None = None

        for index, message in enumerate(messages):
            updated = self._render_tool_message(message)
            if updated is message:
                continue
            if rewritten is None:
                rewritten = list(messages)
            rewritten[index] = updated

        if rewritten is not None:
            state.input.request.messages = rewritten
        context.set_state(state)

    def _render_tool_message(self, message: ChatMessage) -> ChatMessage:
        if message.role != MessageRole.TOOL:
            return message

        kwargs = message.additional_kwargs
        server_result_keys = _server_result_keys(kwargs)
        if not server_result_keys:
            return message

        rendered = _rendered_blocks(kwargs, server_result_keys)
        normalized = normalize_tool_result_content(rendered)
        li_blocks: list[LIContentBlock] = to_llama_index_blocks(normalized)
        projected = _blocks_text(li_blocks)
        if not projected.strip():
            li_blocks.append(TextBlock(text=NO_TOOL_CONTENT).to_llama_index())
            projected = NO_TOOL_CONTENT

        if projected == (message.content or ""):
            return message

        updated = message.model_copy(deep=False)
        updated.additional_kwargs = dict(kwargs)
        updated.blocks = li_blocks
        return updated


def _blocks_text(blocks: list[LIContentBlock]) -> str:
    return "\n".join(
        text
        for block in blocks
        if isinstance((text := getattr(block, "text", None)), str)
    )
