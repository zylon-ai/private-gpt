from __future__ import annotations

from typing import TYPE_CHECKING

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
from private_gpt.events.models._tool_result_blocks import (
    Renderable,
    ServerToolResultBlock,
)

if TYPE_CHECKING:
    from llama_index.core.base.llms.types import ContentBlock as LIContentBlock

    from private_gpt.components.engines.chat.models.chat_interceptor_context import (
        ChatInterceptorContext,
    )


@singleton
class ServerToolResultTextInterceptor(ChatRequestLoopInterceptor):
    """Convert renderable server tool result blocks in the message history to text.

    Walks every ``TOOL`` role message before each iteration, detects any
    ``ServerToolResultBlock`` stored in ``additional_kwargs``, and replaces
    renderable blocks with a plain ``TextBlock`` via ``render()``.
    Non-renderable blocks (e.g. images, source blocks) are preserved as-is.
    """

    async def intercept(self, context: ChatInterceptorContext) -> None:
        if context.phase != InterceptorPhase.BEFORE_ITERATION:
            return

        state = context.state
        messages = state.input.request.messages

        for message in messages:
            if message.role != MessageRole.TOOL:
                continue

            kwargs = message.additional_kwargs
            server_result_keys = [
                key
                for key, value in kwargs.items()
                if isinstance(value, list)
                and any(isinstance(item, ServerToolResultBlock) for item in value)
            ]

            if not server_result_keys:
                continue

            rendered: list = []
            for key in server_result_keys:
                for block in kwargs[key]:
                    if isinstance(block, Renderable):
                        rendered_text = block.render()
                        if rendered_text.strip():
                            rendered.append(TextBlock(text=rendered_text))
                    else:
                        rendered.append(block)
                del kwargs[key]

            normalized = normalize_tool_result_content(rendered)
            li_blocks: list[LIContentBlock] = to_llama_index_blocks(normalized)
            message.blocks = li_blocks or [
                TextBlock(text=NO_TOOL_CONTENT).to_llama_index()
            ]
            message.content = (
                "\n\n".join(
                    block.text for block in normalized if isinstance(block, TextBlock)
                )
                or NO_TOOL_CONTENT
            )

        context.set_state(state)
