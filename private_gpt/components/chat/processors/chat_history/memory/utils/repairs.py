from llama_index.core.base.llms.types import ChatMessage

from private_gpt.components.chat.processors.chat_history.memory.utils.format import (
    guarantee_valid_message_sequence,
)
from private_gpt.components.chat.processors.chat_history.memory.utils.merge import (
    merge_adjacent_messages,
)
from private_gpt.components.chat.processors.chat_history.memory.utils.splitting import (
    get_assistant_tool_pair_mesages,
    get_system_and_conversation_messages,
    get_user_blocks,
)


def repair_without_tools(
    chat_history: list[ChatMessage], strict: bool = True
) -> list[ChatMessage]:
    """Repair the chat history by ensuring it follows the expected structure.

    This method is a reduction of the original chat history
    that it doesn't accept tool messages.
    """
    system_messages, conversation_messages = get_system_and_conversation_messages(
        chat_history
    )

    _assert_user_blocks(conversation_messages, strict)
    _assert_repair_without_tools(conversation_messages)

    conversation_messages = conversation_messages.copy()
    return _repair_conversation(system_messages + conversation_messages)


def repair_with_tools(
    chat_history: list[ChatMessage], strict: bool = True
) -> list[ChatMessage]:
    """Repair the chat history by ensuring it follows the expected structure.

    This method is a reduction of the original chat history
    that it accepts tool messages.

    Only expect a single user block in the chat history.
    """
    system_messages, conversation_messages = get_system_and_conversation_messages(
        chat_history
    )

    _assert_user_blocks(conversation_messages, strict)
    _assert_only_one_user_block(conversation_messages)

    conversation_messages = conversation_messages.copy()
    conversation_messages = _repair_tools(conversation_messages)
    return _repair_conversation(system_messages + conversation_messages)


def _repair_conversation(
    chat_history: list[ChatMessage],
) -> list[ChatMessage]:
    chat_history = chat_history.copy()

    # Merge messages with the same role adjacent to each other
    chat_history = merge_adjacent_messages(chat_history)

    # Guarantee valid user block sequence
    chat_history = guarantee_valid_message_sequence(chat_history)

    return chat_history


def _repair_tools(
    chat_history: list[ChatMessage],
) -> list[ChatMessage]:
    """Repair the chat history by ensuring it follows the expected structure."""
    chat_history = chat_history.copy()
    tool_pairs = get_assistant_tool_pair_mesages(chat_history)
    if not tool_pairs:
        return chat_history

    for assistant_message, tool_message in tool_pairs:
        repaired_messages = _repair_tool_pairs(assistant_message, tool_message)
        if repaired_messages:
            index = chat_history.index(assistant_message)
            chat_history[index : index + 2] = repaired_messages
        else:
            # If the pair is invalid, remove both messages
            chat_history.remove(assistant_message)
            chat_history.remove(tool_message)

    return chat_history


def _repair_tool_pairs(
    assistant_message: ChatMessage,
    tool_message: ChatMessage,
) -> list[ChatMessage]:
    # Validate the assistant message
    assistant_is_valid = True
    if assistant_message.role != "assistant":
        assistant_is_valid = False
    if assistant_message.additional_kwargs.get("tool_calls", None) is None:
        assistant_is_valid = False

    # Validate the tool message
    tool_is_valid = True
    if tool_message.role != "tool":
        tool_is_valid = False
    if tool_message.additional_kwargs.get("tool_call_id", None) is None:
        tool_is_valid = False

    return (
        [assistant_message, tool_message]
        if assistant_is_valid and tool_is_valid
        else []
    )


def _assert_user_blocks(chat_history: list[ChatMessage], strict: bool) -> None:
    """A user block should follow a regex pattern.

    Strict mode: (user, (assistant, tool)*, assistant)
    Non-strict mode: (user, (assistant, tool)*, assistant?)

    """
    if not chat_history:
        return

    matrix = get_user_blocks(chat_history)
    for i, block in enumerate(matrix):
        if not block:
            raise ValueError(f"User block {i} is empty.")

        # Validate the first and last messages in the block
        if block[0].role != "user":
            raise ValueError(
                f"User block {i} does not start with a user message: {block[0]}"
            )

        if len(block) == 1:
            # No more validation needed for a single user message block
            continue

        # Validate the last message in the block
        if strict and block[-1].role != "assistant":
            raise ValueError(
                f"User block {i} does not end with an assistant message: {block[-1]}"
            )
        elif not strict and block[-1].role not in ["assistant", "tool"]:
            raise ValueError(
                f"User block {i} does not end with an assistant or tool message: {block[-1]}"
            )

        # Validate the tool messages
        for j, message in enumerate(block[1:-1]):
            if message.role == "tool":
                if j == 0 or block[j].role != "assistant":
                    raise ValueError(
                        f"Tool message {j} in user block {i} does not follow an assistant message: {message}"
                    )
            elif message.role != "assistant":
                raise ValueError(
                    f"Message {j} in user block {i} is neither an assistant nor a tool: {message}"
                )


def _assert_repair_without_tools(
    chat_history: list[ChatMessage],
) -> None:
    """Assert that the chat history does not contain any tool messages.

    This is used to ensure that the chat history is
    clean and does not contain any tool calls or tool messages.
    """
    if not chat_history:
        return

    def is_a_potential_tool_message(msg: ChatMessage) -> tuple[bool, str | None]:
        if msg.role == "tool":
            return True, "Tool message detected"
        if msg.additional_kwargs.get("tool_calls", None):
            return True, "Tool call detected"
        if msg.additional_kwargs.get("tool_call_id", None):
            return True, "Tool call ID detected"
        return False, None

    for message in chat_history:
        is_tool, reason = is_a_potential_tool_message(message)
        if is_tool:
            raise ValueError(
                f"Chat history contains a tool message: {message.content}. Reason: {reason or 'Unknown'}"
            )


def _assert_only_one_user_block(
    chat_history: list[ChatMessage],
) -> None:
    """Assert that the chat history contains exactly one user block.

    This is used to ensure that the chat history
    is clean and contains only one user block.
    """
    if not chat_history:
        return

    matrix = get_user_blocks(chat_history)
    if not matrix:
        raise ValueError("Chat history does not contain any user messages.")
    if len(matrix) > 1:
        raise ValueError(
            "Chat history contains multiple user blocks. This is not supported."
        )
