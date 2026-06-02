from llama_index.core.base.llms.types import ChatMessage


def get_system_and_conversation_messages(
    chat_history: list[ChatMessage] | None,
) -> tuple[list[ChatMessage], list[ChatMessage]]:
    """Get system and conversation messages from the chat history."""
    if not chat_history:
        return [], []

    system_messages = [msg for msg in chat_history if msg.role == "system"]
    conversation_messages = [msg for msg in chat_history if msg.role != "system"]

    return system_messages, conversation_messages


def get_assistant_tool_pair_mesages(
    chat_history: list[ChatMessage] | None,
) -> list[tuple[ChatMessage, ChatMessage]]:
    """Get all assistant and tool messages from the chat history."""
    pairs: list[tuple[ChatMessage, ChatMessage]] = []
    current_pair: list[ChatMessage] = []

    for message in chat_history or []:
        if message.role == "assistant":
            current_pair.append(message)
        elif message.role == "tool":
            if current_pair:
                pairs.append((current_pair[-1], message))
                current_pair = []

    return pairs


def get_user_blocks(
    chat_history: list[ChatMessage] | None,
) -> list[list[ChatMessage]]:
    """Get all user blocks from the chat history."""
    if not chat_history:
        return []

    blocks: list[list[ChatMessage]] = []
    current_block: list[ChatMessage] = []

    for msg in chat_history.copy():
        if msg.role == "user" and current_block:
            blocks.append(current_block)
            current_block = [msg]
        else:
            current_block.append(msg)

    if current_block:
        blocks.append(current_block)

    return blocks
