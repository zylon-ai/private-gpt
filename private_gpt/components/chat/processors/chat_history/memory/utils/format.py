from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.chat.processors.chat_history.memory.utils.splitting import (
    get_system_and_conversation_messages,
    get_user_blocks,
)


def _guarantee_valid_user_block_sequence(
    messages: list[ChatMessage],
    strict: bool = True,
) -> list[ChatMessage]:
    """Ensure message sequence follows pattern.

    Strict regex: system?, [user, (assistant, tool)*, assistant]+
    Non-strict regex: system?, [user, (assistant, tool)*, assistant?]+
    """
    if not messages:
        return messages

    valid_flags = [False] * len(messages)

    i = 0
    while i < len(messages):
        if messages[i].role == MessageRole.USER:
            valid_flags[i] = True
            i += 1
        elif messages[i].role == MessageRole.ASSISTANT:
            if i + 1 < len(messages) and messages[i + 1].role == MessageRole.TOOL:
                # Pair attempt
                if messages[i].additional_kwargs.get("tool_calls"):
                    valid_flags[i] = True  # Valid assistant
                    valid_flags[i + 1] = True  # Valid tool
                # else: both remain False (invalid pair)
                i += 2
            else:
                valid_flags[i] = True  # Standalone assistant
                i += 1
        else:
            i += 1  # Skip tools and others

    # Collect valid messages based on flags
    valid_messages = [msg for i, msg in enumerate(messages) if valid_flags[i]]

    # Enforce strict mode: sequence must end with assistant
    if strict:
        if valid_messages and (
            valid_messages[-1].role != MessageRole.ASSISTANT
            or valid_messages[-1].additional_kwargs.get("tool_calls")
        ):
            # We drop all user block messages
            # if the last message is not an assistant
            valid_messages = []

    return valid_messages


def guarantee_valid_message_sequence(
    messages: list[ChatMessage],
) -> list[ChatMessage]:
    """Ensure message sequence follows pattern.

    [0, N): [user, (assistant, tool)*, assistant]+
    [N: N]: [user, (assistant, tool)*, assistant?]+
    """
    if not messages:
        return []

    final_messages: list[ChatMessage] = []

    system_messages, conversation_messages = get_system_and_conversation_messages(
        messages
    )
    blocks = get_user_blocks(conversation_messages)
    for i, interaction in enumerate(blocks):
        if not interaction:
            continue

        strict = i != len(blocks) - 1  # Last block is non-strict
        new_interaction = _guarantee_valid_user_block_sequence(
            interaction, strict=strict
        )
        final_messages.extend(new_interaction)

    return system_messages + _guarantee_valid_user_block_sequence(
        final_messages, strict=False
    )
