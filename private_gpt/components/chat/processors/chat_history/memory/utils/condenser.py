from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.chat.processors.chat_history.memory.utils.content import (
    messages_to_history_str,
)
from private_gpt.components.chat.processors.chat_history.memory.utils.format import (
    guarantee_valid_message_sequence,
)
from private_gpt.events.models import BasicContentBlockType, TextBlock


def build_condensed_history(
    system_messages: list[ChatMessage],
    conversation_history: list[ChatMessage],
) -> tuple[list[ChatMessage], list[BasicContentBlockType]]:
    """Build the final condensed chat history."""
    last_user_message: ChatMessage | None = None
    for msg in reversed(conversation_history):
        if msg.role == MessageRole.USER:
            last_user_message = msg
            break

    if not last_user_message:
        raise ValueError("No user message found in conversation history.")

    def is_truncated(msg: ChatMessage) -> bool:
        if not msg.additional_kwargs:
            return False
        if "tldr" not in msg.additional_kwargs:
            return False
        tldr_value = msg.additional_kwargs["tldr"]
        return bool(tldr_value)

    truncated_blocks: list[BasicContentBlockType] = [
        TextBlock(
            text=messages_to_history_str([msg], show_index=False, show_role=False),
            metadata={
                "type": "tldr",
                "role": msg.role,
                "tldr_side": msg.additional_kwargs.get("tldr", "left")
                if isinstance(msg.additional_kwargs.get("tldr"), str)
                else "left",
            },
        )
        for msg in conversation_history
        if is_truncated(msg)
    ]

    condensed_history = system_messages + conversation_history
    return (
        guarantee_valid_message_sequence(condensed_history),
        truncated_blocks,
    )
