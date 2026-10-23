from llama_index.core.base.llms.types import ChatMessage


def merge_adjacent_messages(
    chat_history: list[ChatMessage],
) -> list[ChatMessage]:
    """Merge adjacent messages with the same role."""
    if not chat_history:
        return []

    merged_history: list[ChatMessage] = []
    last_message: ChatMessage | None = None

    for message in chat_history:
        if last_message and message.role == last_message.role:
            last_message.content = "\n".join(
                [msg.content for msg in [last_message, message] if msg.content]
            )
            last_message.additional_kwargs.update(message.additional_kwargs)
        else:
            if last_message:
                merged_history.append(last_message)
            last_message = message

    if last_message:
        merged_history.append(last_message)

    return merged_history
