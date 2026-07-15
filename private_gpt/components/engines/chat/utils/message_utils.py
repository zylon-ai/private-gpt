from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock


def latest_user_text(messages: list[ChatMessage]) -> str:
    """Return the last user text content from conversation."""
    for message in reversed(messages):
        if message.role != MessageRole.USER:
            continue
        parts: list[str] = []
        for block in message.blocks:
            if isinstance(block, TextBlock) and block.text:
                parts.append(block.text)
        if parts:
            return "\n".join(parts)
    return ""


def upsert_system_message(
    messages: list[ChatMessage], content: str
) -> list[ChatMessage]:
    """Set one system message at top of conversation and return new list."""
    without_system = [m for m in messages if m.role != MessageRole.SYSTEM]
    system_message = ChatMessage(role=MessageRole.SYSTEM, content=content)
    return [system_message, *without_system]
