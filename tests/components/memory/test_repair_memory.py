from typing import Any

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.chat.processors.chat_history.memory.utils.repairs import (
    repair_with_tools,
    repair_without_tools,
)


@pytest.fixture
def system_message() -> ChatMessage:
    return ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant")


@pytest.fixture
def user_message() -> ChatMessage:
    return ChatMessage(role=MessageRole.USER, content="Hello, can you help me?")


@pytest.fixture
def user_message_2() -> ChatMessage:
    return ChatMessage(role=MessageRole.USER, content="Another question")


@pytest.fixture
def assistant_message() -> ChatMessage:
    return ChatMessage(role=MessageRole.ASSISTANT, content="I'll help you with that")


@pytest.fixture
def assistant_message_2() -> ChatMessage:
    return ChatMessage(role=MessageRole.ASSISTANT, content="Here's another response")


@pytest.fixture
def assistant_with_tool_calls() -> ChatMessage:
    return ChatMessage(
        role=MessageRole.ASSISTANT,
        content="I'll search for information",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {
                        "name": "search_tool",
                        "arguments": '{"query": "test"}',
                    },
                    "type": "function",
                }
            ]
        },
    )


@pytest.fixture
def tool_message() -> ChatMessage:
    return ChatMessage(
        role=MessageRole.TOOL,
        content="Search results found",
        additional_kwargs={
            "tool_call_id": "call_123",
            "tool_call_name": "search_tool",
            "raw_output": "Search results found",
        },
    )


@pytest.fixture
def invalid_tool_message() -> ChatMessage:
    return ChatMessage(
        role=MessageRole.TOOL,
        content="Invalid tool result",
        additional_kwargs={
            "tool_call_name": "invalid_tool"
            # Missing tool_call_id
        },
    )


# Tests for repair_without_tools


async def test_repair_without_tools_empty_input() -> None:
    result: list[ChatMessage] = repair_without_tools([])
    assert result == []


@pytest.mark.parametrize(
    ("strict", "should_succeed"),
    [
        (True, True),
        (False, True),
    ],
)
async def test_repair_without_tools_valid_simple_conversation(
    user_message: ChatMessage,
    assistant_message: ChatMessage,
    strict: bool,
    should_succeed: bool,
) -> None:
    messages: list[ChatMessage] = [user_message, assistant_message]
    result: list[ChatMessage] = repair_without_tools(messages, strict)

    assert len(result) == 2
    assert result[0].role == MessageRole.USER
    assert result[1].role == MessageRole.ASSISTANT


async def test_repair_without_tools_with_system_message(
    system_message: ChatMessage,
    user_message: ChatMessage,
    assistant_message: ChatMessage,
) -> None:
    messages: list[ChatMessage] = [system_message, user_message, assistant_message]
    result: list[ChatMessage] = repair_without_tools(messages)

    assert len(result) >= 3
    system_messages: list[ChatMessage] = [
        msg for msg in result if msg.role == MessageRole.SYSTEM
    ]
    assert len(system_messages) == 1


async def test_repair_without_tools_multiple_user_blocks_strict(
    user_message: ChatMessage,
    user_message_2: ChatMessage,
    assistant_message: ChatMessage,
    assistant_message_2: ChatMessage,
) -> None:
    messages: list[ChatMessage] = [
        user_message,
        assistant_message,
        user_message_2,
        assistant_message_2,
    ]
    result: list[ChatMessage] = repair_without_tools(messages, strict=True)

    # Should preserve all messages since each user block ends with assistant
    user_messages: list[ChatMessage] = [
        msg for msg in result if msg.role == MessageRole.USER
    ]
    assistant_messages: list[ChatMessage] = [
        msg for msg in result if msg.role == MessageRole.ASSISTANT
    ]

    assert len(user_messages) == 2
    assert len(assistant_messages) == 2


@pytest.mark.parametrize(
    ("strict", "expected_min_length"),
    [
        (True, 2),  # Should end with assistant in strict mode
        (False, 2),  # Should preserve incomplete block in non-strict mode
    ],
)
async def test_repair_without_tools_incomplete_user_block(
    user_message: ChatMessage,
    user_message_2: ChatMessage,
    assistant_message: ChatMessage,
    strict: bool,
    expected_min_length: int,
) -> None:
    messages: list[ChatMessage] = [
        user_message,
        assistant_message,
        user_message_2,  # Incomplete block - no assistant after this
    ]
    result: list[ChatMessage] = repair_without_tools(messages, strict)

    assert len(result) >= expected_min_length
    user_messages: list[ChatMessage] = [
        msg for msg in result if msg.role == MessageRole.USER
    ]
    assert len(user_messages) >= 1


async def test_repair_without_tools_merges_adjacent_same_role(
    user_message: ChatMessage,
    assistant_message: ChatMessage,
    assistant_message_2: ChatMessage,
) -> None:
    # Create adjacent assistant messages that should be merged
    messages: list[ChatMessage] = [
        user_message,
        assistant_message,
        assistant_message_2,  # Adjacent assistant
    ]
    result: list[ChatMessage] = repair_without_tools(messages)

    # Should merge adjacent assistant messages
    assistant_messages: list[ChatMessage] = [
        msg for msg in result if msg.role == MessageRole.ASSISTANT
    ]
    # After merging, should have fewer assistant messages than input
    assert len(assistant_messages) <= 2


async def test_repair_without_tools_raises_on_tool_messages(
    user_message: ChatMessage,
    assistant_message: ChatMessage,
    tool_message: ChatMessage,
) -> None:
    messages: list[ChatMessage] = [user_message, assistant_message, tool_message]

    with pytest.raises(ValueError, match="Tool message detected"):
        repair_without_tools(messages, strict=False)


async def test_repair_without_tools_raises_on_tool_calls(
    user_message: ChatMessage,
    assistant_with_tool_calls: ChatMessage,
) -> None:
    messages: list[ChatMessage] = [user_message, assistant_with_tool_calls]

    with pytest.raises(ValueError, match="Tool call detected"):
        repair_without_tools(messages)


# Tests for repair_with_tools


async def test_repair_with_tools_empty_input() -> None:
    result: list[ChatMessage] = repair_with_tools([])
    assert result == []


async def test_repair_with_tools_simple_conversation_no_tools(
    user_message: ChatMessage,
    assistant_message: ChatMessage,
) -> None:
    messages: list[ChatMessage] = [user_message, assistant_message]
    result: list[ChatMessage] = repair_with_tools(messages)

    assert len(result) == 2
    assert result[0].role == MessageRole.USER
    assert result[1].role == MessageRole.ASSISTANT


async def test_repair_with_tools_valid_tool_pair(
    user_message: ChatMessage,
    assistant_with_tool_calls: ChatMessage,
    tool_message: ChatMessage,
    assistant_message: ChatMessage,
) -> None:
    messages: list[ChatMessage] = [
        user_message,
        assistant_with_tool_calls,
        tool_message,
        assistant_message,
    ]
    result: list[ChatMessage] = repair_with_tools(messages)

    assert len(result) == 4
    assert result[0].role == MessageRole.USER
    assert result[1].role == MessageRole.ASSISTANT
    assert result[2].role == MessageRole.TOOL
    assert result[3].role == MessageRole.ASSISTANT


async def test_repair_with_tools_invalid_tool_pair_removed(
    user_message: ChatMessage,
    assistant_message: ChatMessage,  # No tool_calls
    tool_message: ChatMessage,
    assistant_message_2: ChatMessage,
) -> None:
    messages: list[ChatMessage] = [
        user_message,
        assistant_message,  # Assistant without tool_calls
        tool_message,  # Tool without matching tool_calls
        assistant_message_2,
    ]
    result: list[ChatMessage] = repair_with_tools(messages)

    # Invalid pair should be removed
    assert len(result) == 2
    assert result[0].role == MessageRole.USER
    assert result[1].role == MessageRole.ASSISTANT
    assert result[1] == assistant_message_2


async def test_repair_with_tools_multiple_valid_tool_pairs(
    user_message: ChatMessage,
    assistant_message: ChatMessage,
) -> None:
    assistant_with_tools_1: ChatMessage = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="First tool call",
        additional_kwargs={
            "tool_calls": [{"id": "call_1", "function": {"name": "tool1"}}]
        },
    )
    tool_1: ChatMessage = ChatMessage(
        role=MessageRole.TOOL,
        content="Tool 1 result",
        additional_kwargs={"tool_call_id": "call_1"},
    )
    assistant_with_tools_2: ChatMessage = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="Second tool call",
        additional_kwargs={
            "tool_calls": [{"id": "call_2", "function": {"name": "tool2"}}]
        },
    )
    tool_2: ChatMessage = ChatMessage(
        role=MessageRole.TOOL,
        content="Tool 2 result",
        additional_kwargs={"tool_call_id": "call_2"},
    )

    messages: list[ChatMessage] = [
        user_message,
        assistant_with_tools_1,
        tool_1,
        assistant_with_tools_2,
        tool_2,
        assistant_message,
    ]
    result: list[ChatMessage] = repair_with_tools(messages)

    assert len(result) == 6
    tool_messages: list[ChatMessage] = [
        msg for msg in result if msg.role == MessageRole.TOOL
    ]
    assert len(tool_messages) == 2


async def test_repair_with_tools_raises_on_multiple_user_blocks(
    user_message: ChatMessage,
    user_message_2: ChatMessage,
    assistant_message: ChatMessage,
    assistant_message_2: ChatMessage,
) -> None:
    messages: list[ChatMessage] = [
        user_message,
        assistant_message,
        user_message_2,  # Second user block not allowed
        assistant_message_2,
    ]

    with pytest.raises(ValueError, match="multiple user blocks"):
        repair_with_tools(messages)


@pytest.mark.parametrize(
    ("strict", "should_succeed"),
    [
        (True, False),  # Strict mode requires ending with assistant
        (False, True),  # Non-strict mode allows ending with tool
    ],
)
async def test_repair_with_tools_incomplete_tool_sequence(
    user_message: ChatMessage,
    assistant_with_tool_calls: ChatMessage,
    tool_message: ChatMessage,
    strict: bool,
    should_succeed: bool,
) -> None:
    messages: list[ChatMessage] = [
        user_message,
        assistant_with_tool_calls,
        tool_message,  # Ends with tool, no final assistant
    ]

    if should_succeed:
        result: list[ChatMessage] = repair_with_tools(messages, strict)
        assert len(result) >= 1
    else:
        with pytest.raises(ValueError, match="does not end with an assistant"):
            repair_with_tools(messages, strict)


async def test_repair_with_tools_merges_adjacent_messages(
    user_message: ChatMessage,
    assistant_message: ChatMessage,
    assistant_message_2: ChatMessage,
) -> None:
    # Two adjacent assistant messages should be merged
    messages: list[ChatMessage] = [
        user_message,
        assistant_message,
        assistant_message_2,  # Adjacent to previous assistant
    ]
    result: list[ChatMessage] = repair_with_tools(messages)

    # Should have merged the adjacent assistants
    assistant_messages: list[ChatMessage] = [
        msg for msg in result if msg.role == MessageRole.ASSISTANT
    ]
    assert len(assistant_messages) == 1
    # Merged message should contain content from both
    merged_content: str = "".join(
        [block.text for block in assistant_messages[0].blocks]
    )
    assert "I'll help you with that" in merged_content


async def test_repair_with_tools_system_messages_preserved(
    system_message: ChatMessage,
    user_message: ChatMessage,
    assistant_message: ChatMessage,
) -> None:
    messages: list[ChatMessage] = [system_message, user_message, assistant_message]
    result: list[ChatMessage] = repair_with_tools(messages)

    system_messages: list[ChatMessage] = [
        msg for msg in result if msg.role == MessageRole.SYSTEM
    ]
    assert len(system_messages) == 1
    assert system_messages[0].content == "You are a helpful assistant"


@pytest.mark.parametrize(
    ("has_tool_calls", "has_tool_call_id", "should_be_valid"),
    [
        (True, True, True),  # Valid pair
        (True, False, False),  # Assistant has tool_calls but tool lacks ID
        (False, True, False),  # Assistant lacks tool_calls but tool has ID
        (False, False, False),  # Neither has required fields
    ],
)
async def test_repair_with_tools_tool_pair_validation(
    user_message: ChatMessage,
    assistant_message: ChatMessage,
    has_tool_calls: bool,
    has_tool_call_id: bool,
    should_be_valid: bool,
) -> None:
    # Create assistant with or without tool_calls
    assistant_kwargs: dict[str, Any] = {}
    if has_tool_calls:
        assistant_kwargs["tool_calls"] = [
            {"id": "call_123", "function": {"name": "test"}}
        ]

    assistant: ChatMessage = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="Assistant message",
        additional_kwargs=assistant_kwargs,
    )

    # Create tool with or without tool_call_id
    tool_kwargs: dict[str, Any] = {}
    if has_tool_call_id:
        tool_kwargs["tool_call_id"] = "call_123"

    tool: ChatMessage = ChatMessage(
        role=MessageRole.TOOL, content="Tool result", additional_kwargs=tool_kwargs
    )

    messages: list[ChatMessage] = [
        user_message,
        assistant,
        tool,
        assistant_message,  # Final assistant
    ]

    result: list[ChatMessage] = repair_with_tools(messages)

    if should_be_valid:
        # Valid pair should be preserved
        tool_messages: list[ChatMessage] = [
            msg for msg in result if msg.role == MessageRole.TOOL
        ]
        assert len(tool_messages) == 1
    else:
        # Invalid pair should be removed
        tool_messages = [msg for msg in result if msg.role == MessageRole.TOOL]
        assert len(tool_messages) == 0


async def test_repair_with_tools_preserves_message_order(
    user_message: ChatMessage,
    assistant_with_tool_calls: ChatMessage,
    tool_message: ChatMessage,
    assistant_message: ChatMessage,
) -> None:
    messages: list[ChatMessage] = [
        user_message,
        assistant_with_tool_calls,
        tool_message,
        assistant_message,
    ]
    result: list[ChatMessage] = repair_with_tools(messages)

    # Verify the order is preserved
    assert result[0].role == MessageRole.USER
    assert result[1].role == MessageRole.ASSISTANT
    assert result[1].additional_kwargs.get("tool_calls") is not None
    assert result[2].role == MessageRole.TOOL
    assert result[3].role == MessageRole.ASSISTANT
    assert result[3].additional_kwargs.get("tool_calls") is None
