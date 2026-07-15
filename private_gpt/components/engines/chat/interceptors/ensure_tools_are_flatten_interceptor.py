from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import ToolSelection

from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import (
    InterceptorPhase,
)


class EnsureToolAreFlattenInterceptor(ChatRequestLoopInterceptor):
    async def intercept(self, context: ChatInterceptorContext) -> None:
        if context.phase != InterceptorPhase.AFTER_ITERATION:
            return

        context.state.input.request.messages = self._reorder_tool_messages(
            context.state.input.request.messages
        )
        if context.state.original_input:
            context.state.original_input.request.messages = self._reorder_tool_messages(
                context.state.original_input.request.messages
            )

        context.set_state(context.state)

    def _reorder_tool_messages(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Reorder to ensure correct messages orders.

        This is necessary because when the model generate several tools calls in the
        same assistant message, we only have one assistant message with multiple tools
        and this broke the TLDR.
        """
        new_messages: list[ChatMessage] = []
        tool_messages_look_up: dict[str, ChatMessage] = {
            msg.additional_kwargs.get("tool_call_id", ""): msg
            for msg in messages
            if msg.role == MessageRole.TOOL
        }
        for message in messages:
            if message.role == MessageRole.ASSISTANT:
                tool_calls = message.additional_kwargs.get("tool_calls", [])
                if tool_calls and isinstance(tool_calls, list):
                    found_assistant = False
                    for tool_call_data in tool_calls:
                        if isinstance(tool_call_data, ToolSelection):
                            assistant_msg = message.model_copy(deep=True)
                            if found_assistant:
                                assistant_msg.additional_kwargs = {}
                            assistant_msg.additional_kwargs = {
                                "tool_calls": [tool_call_data],
                            }
                            if "tldr" in message.additional_kwargs:
                                assistant_msg.additional_kwargs["tldr"] = (
                                    message.additional_kwargs["tldr"]
                                )

                            found_assistant = True

                            tool_msg = tool_messages_look_up.get(tool_call_data.tool_id)
                            if not tool_msg:
                                raise ValueError(
                                    f"Tool message with id {tool_call_data.tool_id} "
                                    f"not found for assistant message {message}"
                                )

                            new_messages.append(assistant_msg)
                            new_messages.append(tool_msg)
                    continue
            elif message.role == MessageRole.TOOL:
                continue

            new_messages.append(message)

        return new_messages
