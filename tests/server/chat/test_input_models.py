import asyncio
from typing import Any

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.llms.types import (
    TextBlock as LITextBlock,
)
from llama_index.core.schema import NodeWithScore, TextNode
from pydantic import ValidationError

from private_gpt.chat.input_models import MessageInput, ToolSpecBody
from private_gpt.components.chunk.models import Chunk
from private_gpt.components.engines.citations.utils import process_history_citations
from private_gpt.events.models import (
    ImageBlock,
    MidConvSystemBlock,
    SourceBlock,
    TextBlock,
    ThinkingBlock,
    TLDRBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def test_anthropic_message_default_role() -> None:
    with pytest.raises(ValidationError):
        MessageInput(content="test")


@pytest.mark.parametrize(
    ("role", "content", "expected_role", "expected_content"),
    [
        ("user", "Hello", MessageRole.USER, "Hello"),
        ("assistant", "Response", MessageRole.ASSISTANT, "Response"),
    ],
)
def test_simple_message_conversion(
    role: str, content: str, expected_role: MessageRole, expected_content: str
) -> None:
    message = MessageInput(role=role, content=content)  # type: ignore[arg-type]
    result, tool_uses = message._convert_into_llama_index_messages()

    assert len(result) == 1
    assert result[0].role == expected_role
    assert result[0].content == expected_content
    assert tool_uses == {}


def test_user_message_with_mid_conv_system_block() -> None:
    message = MessageInput(
        role="user",
        content=[
            TextBlock(type="text", text="Hello"),
            MidConvSystemBlock(
                content=[
                    TextBlock(type="text", text="Be concise."),
                    TextBlock(type="text", text="Use bullet points."),
                ]
            ),
        ],
    )
    result, _ = message._convert_into_llama_index_messages()

    assert len(result) == 1
    assert result[0].role == MessageRole.USER
    blocks = result[0].blocks
    assert len(blocks) == 2
    assert isinstance(blocks[0], LITextBlock)
    assert blocks[0].text == "Hello"
    assert isinstance(blocks[1], LITextBlock)
    assert blocks[1].text == "Be concise.\nUse bullet points."


def test_user_message_mid_conv_system_single_text_block() -> None:
    message = MessageInput(
        role="user",
        content=[
            MidConvSystemBlock(
                content=[TextBlock(type="text", text="Focus on safety.")]
            ),
        ],
    )
    result, _ = message._convert_into_llama_index_messages()

    assert len(result) == 1
    blocks = result[0].blocks
    assert len(blocks) == 1
    assert isinstance(blocks[0], LITextBlock)
    assert blocks[0].text == "Focus on safety."


def test_assistant_message_with_text_block() -> None:
    text_block = TextBlock(type="text", text="Hello from assistant")
    message = MessageInput(role="assistant", content=[text_block])

    result, tool_uses = message._convert_into_llama_index_messages()

    assert len(result) == 1
    assert result[0].role == MessageRole.ASSISTANT
    assert result[0].content == "Hello from assistant"
    assert tool_uses == {}


def test_assistant_message_with_tool_blocks() -> None:
    text_block = TextBlock(type="text", text="Using tools")
    tool_block1 = ToolUseBlock(
        type="tool_use",
        id="tool1",
        name="calculator",
        input={"operation": "add", "x": 1, "y": 2},
    )
    tool_block2 = ToolUseBlock(
        type="tool_use", id="tool2", name="search", input={"query": "weather"}
    )

    message = MessageInput(
        role="assistant", content=[text_block, tool_block1, tool_block2]
    )

    result, tool_uses = message._convert_into_llama_index_messages()

    assert len(result) == 2

    # First message has text content and first tool
    assert result[0].role == MessageRole.ASSISTANT
    assert result[0].content == "Using tools"
    assert len(result[0].additional_kwargs["tool_calls"]) == 1
    tool_call1 = result[0].additional_kwargs["tool_calls"][0]
    assert tool_call1.tool_id == "tool1"
    assert tool_call1.tool_name == "calculator"
    assert tool_call1.tool_kwargs == {"operation": "add", "x": 1, "y": 2}

    # Second message has no text content and second tool
    assert result[1].role == MessageRole.ASSISTANT
    assert result[1].content is None
    assert len(result[1].additional_kwargs["tool_calls"]) == 1
    tool_call2 = result[1].additional_kwargs["tool_calls"][0]
    assert tool_call2.tool_id == "tool2"
    assert tool_call2.tool_name == "search"
    assert tool_call2.tool_kwargs == {"query": "weather"}

    # Check tool_uses mapping
    assert len(tool_uses) == 2
    assert tool_uses["tool1"] == tool_block1
    assert tool_uses["tool2"] == tool_block2


def test_tool_message_with_result_blocks() -> None:
    result_text = TextBlock(type="text", text="Result: 3")
    tool_result = ToolResultBlock(
        type="tool_result", tool_use_id="tool1", content=[result_text]
    )

    message = MessageInput(role="assistant", content=[tool_result])

    # Mock the tool_uses to simulate previous tool usage
    tool_uses = {
        "tool1": ToolUseBlock(type="tool_use", id="tool1", name="calculator", input={})
    }

    result, _ = message._convert_into_llama_index_messages(tool_uses)

    assert len(result) == 1
    assert result[0].role == MessageRole.TOOL
    assert result[0].content == "Result: 3"
    assert result[0].additional_kwargs["tool_call_id"] == "tool1"
    assert result[0].additional_kwargs["tool_call_name"] == "calculator"


def test_tool_result_with_empty_content() -> None:
    tool_result = ToolResultBlock(type="tool_result", tool_use_id="tool1", content=[])

    message = MessageInput(role="assistant", content=[tool_result])

    tool_uses = {
        "tool1": ToolUseBlock(type="tool_use", id="tool1", name="calculator", input={})
    }

    result, _ = message._convert_into_llama_index_messages(tool_uses)

    assert len(result) == 1
    assert result[0].role == MessageRole.TOOL
    assert result[0].content == "No content"
    assert result[0].additional_kwargs["tool_call_id"] == "tool1"
    assert result[0].additional_kwargs["tool_call_name"] == "calculator"


def test_extract_text_content_with_various_inputs() -> None:
    message = MessageInput(role="user", content="test")

    # Test with string
    blocks, custom_blocks = message._extract_content("Hello")
    assert blocks is not None
    text_block = next(
        (b for b in blocks if isinstance(b, LITextBlock)),
        None,
    )
    assert text_block.text == "Hello"

    # Test with TextBlock in list
    text_block = TextBlock(type="text", text="Block text")
    blocks, custom_blocks = message._extract_content([text_block])
    assert blocks is not None
    text_block = next(
        (b for b in blocks if isinstance(b, LITextBlock)),
        None,
    )
    assert text_block.text == "Block text"

    # Test with non-text blocks
    tool_block = ToolUseBlock(type="tool_use", id="1", name="tool", input={})
    blocks, custom_blocks = message._extract_content([tool_block])
    assert blocks is None
    assert custom_blocks is not None
    assert len(custom_blocks.items()) == 1

    # Test with None
    blocks, custom_blocks = message._extract_content(None)
    assert blocks is None
    assert custom_blocks is None


def test_convert_from_llama_index_messages_with_reordering() -> None:
    # Create messages that need reordering
    messages = [
        MessageInput(role="user", content="Calculate 2+2"),
        MessageInput(
            role="assistant",
            content=[
                TextBlock(type="text", text="Let me calculate"),
                ToolUseBlock(
                    type="tool_use",
                    id="calc1",
                    name="calculator",
                    input={"operation": "add"},
                ),
                ToolUseBlock(
                    type="tool_use",
                    id="calc2",
                    name="calculator",
                    input={"operation": "add"},
                ),
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="calc1",
                    content=[TextBlock(type="text", text="4")],
                ),
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="calc2",
                    content=[TextBlock(type="text", text="4")],
                ),
            ],
        ),
    ]

    result = MessageInput.convert_from_llama_index_messages(messages)

    # Check that tool response is moved after its assistant message
    assert result[0].role == MessageRole.USER
    assert result[1].role == MessageRole.ASSISTANT
    assert result[2].role == MessageRole.TOOL  # Moved from position 3
    assert result[3].role == MessageRole.ASSISTANT
    assert result[4].role == MessageRole.TOOL

    # Verify tool connection
    assert result[1].additional_kwargs["tool_calls"][0].tool_id == "calc1"
    assert result[2].additional_kwargs["tool_call_id"] == "calc1"


@pytest.mark.parametrize(
    ("name", "description", "input_schema"),
    [
        ("calculator", "Performs calculations", {"type": "object", "properties": {}}),
        ("search", None, {"type": "object", "required": ["query"]}),
        (
            "data_fetcher",
            "Fetches data",
            {"type": "object", "properties": {"_id": {"type": "string"}}},
        ),
    ],
)
def test_anthropic_tool_creation(
    name: str, description: str | None, input_schema: dict[str, Any]
) -> None:
    tool = ToolSpecBody(name=name, description=description, input_schema=input_schema)

    assert tool.name == name
    assert tool.description == description
    assert tool.input_schema == input_schema


def test_empty_content_handling() -> None:
    with pytest.raises(ValidationError):
        MessageInput(role="assistant", content=None)


def test_complex_tool_workflow() -> None:
    # Simulate a complete workflow with tools
    messages = [
        MessageInput(role="user", content="What's the weather?"),
        MessageInput(
            role="assistant",
            content=[
                TextBlock(type="text", text="I'll check the weather"),
                ToolUseBlock(
                    type="tool_use",
                    id="weather1",
                    name="weather_api",
                    input={"location": "New York"},
                ),
            ],
        ),
        MessageInput(
            role="assistant",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="weather1",
                    content=[TextBlock(type="text", text="Sunny, 72°F")],
                )
            ],
        ),
        MessageInput(
            role="assistant", content="The weather in New York is sunny and 72°F."
        ),
    ]

    result = MessageInput.convert_from_llama_index_messages(messages)

    # Verify the flow
    assert len(result) == 4
    assert result[0].role == MessageRole.USER
    assert result[1].role == MessageRole.ASSISTANT
    assert result[1].additional_kwargs["tool_calls"][0].tool_name == "weather_api"
    assert result[2].role == MessageRole.TOOL
    assert result[2].content == "Sunny, 72°F"
    assert result[3].role == MessageRole.ASSISTANT
    assert result[3].content == "The weather in New York is sunny and 72°F."


def test_multiple_tools_same_message() -> None:
    messages = [
        MessageInput(
            role="user",
            content="What's the weather in NYC and search for restaurants there?",
        ),
        MessageInput(
            role="assistant",
            content=[
                TextBlock(
                    type="text", text="I'll check the weather and find restaurants"
                ),
                ToolUseBlock(
                    type="tool_use",
                    id="weather1",
                    name="weather_api",
                    input={"location": "New York"},
                ),
                ToolUseBlock(
                    type="tool_use",
                    id="search1",
                    name="search_api",
                    input={"query": "restaurants in New York", "type": "local"},
                ),
                ToolUseBlock(
                    type="tool_use",
                    id="maps1",
                    name="maps_api",
                    input={"location": "New York", "category": "restaurants"},
                ),
            ],
        ),
        MessageInput(
            role="assistant",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="weather1",
                    content=[TextBlock(type="text", text="Sunny, 72°F")],
                )
            ],
        ),
        MessageInput(
            role="assistant",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="search1",
                    content=[
                        TextBlock(
                            type="text",
                            text="Found 50 restaurants: Italian, Chinese, Mexican...",
                        )
                    ],
                )
            ],
        ),
        MessageInput(
            role="assistant",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="maps1",
                    content=[
                        TextBlock(
                            type="text",
                            text="Top rated: Joe's Pizza (4.5★), Dragon Palace (4.7★)",
                        )
                    ],
                )
            ],
        ),
        MessageInput(
            role="assistant",
            content="The weather in NYC is sunny and 72°F. I found many restaurants including Joe's Pizza and Dragon Palace which are highly rated.",
        ),
    ]

    result = MessageInput.convert_from_llama_index_messages(messages)

    # Should create 3 separate assistant messages for each tool
    assert len(result) == 8  # user + 3*(assistant+tool) + final assistant

    # Check user message
    assert result[0].role == MessageRole.USER

    # Check first tool (weather)
    assert result[1].role == MessageRole.ASSISTANT
    assert result[1].content == "I'll check the weather and find restaurants"
    assert result[1].additional_kwargs["tool_calls"][0].tool_name == "weather_api"
    assert result[2].role == MessageRole.TOOL
    assert result[2].additional_kwargs["tool_call_id"] == "weather1"

    # Check second tool (search)
    assert result[3].role == MessageRole.ASSISTANT
    assert result[3].content is None  # Only first tool has text content
    assert result[3].additional_kwargs["tool_calls"][0].tool_name == "search_api"
    assert result[4].role == MessageRole.TOOL
    assert result[4].additional_kwargs["tool_call_id"] == "search1"

    # Check third tool (maps)
    assert result[5].role == MessageRole.ASSISTANT
    assert result[5].content is None
    assert result[5].additional_kwargs["tool_calls"][0].tool_name == "maps_api"
    assert result[6].role == MessageRole.TOOL
    assert result[6].additional_kwargs["tool_call_id"] == "maps1"

    # Check final assistant response
    assert result[7].role == MessageRole.ASSISTANT
    assert "sunny and 72°F" in result[7].content
    assert "Joe's Pizza" in result[7].content


def test_multiple_tools_with_followups() -> None:
    messages = [
        MessageInput(role="user", content="Plan my trip to Tokyo"),
        MessageInput(
            role="assistant",
            content=[
                TextBlock(
                    type="text",
                    text="I'll help plan your trip. Let me check flights and weather",
                ),
                ToolUseBlock(
                    type="tool_use",
                    id="flight1",
                    name="flight_search",
                    input={"destination": "Tokyo", "dates": "flexible"},
                ),
                ToolUseBlock(
                    type="tool_use",
                    id="weather1",
                    name="weather_api",
                    input={"location": "Tokyo", "forecast": "7days"},
                ),
            ],
        ),
        MessageInput(
            role="assistant",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="flight1",
                    content=[
                        TextBlock(
                            type="text", text="Best flights: $800-1200, 12-14 hours"
                        )
                    ],
                )
            ],
        ),
        MessageInput(
            role="assistant",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="weather1",
                    content=[
                        TextBlock(type="text", text="Next week: Partly cloudy, 18-22°C")
                    ],
                )
            ],
        ),
        # Second round - split into two messages again
        MessageInput(
            role="assistant",
            content=[
                TextBlock(
                    type="text",
                    text="Great! Flights are $800-1200. Weather looks good. Let me find hotels and attractions",
                ),
                ToolUseBlock(
                    type="tool_use",
                    id="hotel1",
                    name="hotel_search",
                    input={"city": "Tokyo", "budget": "moderate"},
                ),
            ],
        ),
        MessageInput(
            role="assistant",
            content=[
                ToolUseBlock(
                    type="tool_use",
                    id="attraction1",
                    name="attraction_search",
                    input={"city": "Tokyo", "type": "popular"},
                )
            ],
        ),
        # Tool results can be combined in a single message
        MessageInput(
            role="user",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="hotel1",
                    content=[
                        TextBlock(
                            type="text",
                            text="Hotels: Shinjuku Grand ($150/night), Tokyo Bay ($120/night)",
                        ),
                        SourceBlock(
                            type="source",
                            sources=[
                                Chunk.from_node(
                                    NodeWithScore(node=TextNode(text="Shinjuku Grand"))
                                ),
                                Chunk.from_node(
                                    NodeWithScore(node=TextNode(text="Tokyo Bay"))
                                ),
                            ],
                        ),
                    ],
                ),
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="attraction1",
                    content=[
                        TextBlock(
                            type="text",
                            text="Must see: Senso-ji Temple, Tokyo Skytree, Shibuya Crossing",
                        ),
                        SourceBlock(
                            type="source",
                            sources=[
                                Chunk.from_node(
                                    NodeWithScore(node=TextNode(text="Shinjuku Grand"))
                                ),
                                Chunk.from_node(
                                    NodeWithScore(node=TextNode(text="Tokyo Bay"))
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        MessageInput(
            role="assistant",
            content="Your Tokyo trip is planned! Flights: $800-1200, Weather: 18-22°C partly cloudy."
            "Hotels: Shinjuku Grand or Tokyo Bay. Must-see: Senso-ji Temple, Tokyo Skytree, and Shibuya Crossing.",
        ),
    ]

    result = MessageInput.convert_from_llama_index_messages(messages)

    # Count messages: user + 2 pairs (assistant+tool)
    #                 + 2 pairs (assistant+tool) + final assistant
    assert len(result) == 10

    # After reordering: user, assistant1, tool1, assistant2, tool2,
    #                   assistant3, tool3, assistant4, tool4, final assistant
    assert result[0].role == MessageRole.USER

    # First assistant+tool pair
    assert result[1].role == MessageRole.ASSISTANT
    assert (
        result[1].content
        == "I'll help plan your trip. Let me check flights and weather"
    )
    assert result[1].additional_kwargs["tool_calls"][0].tool_name == "flight_search"
    assert result[2].role == MessageRole.TOOL
    assert result[2].additional_kwargs["tool_call_id"] == "flight1"

    # Second assistant+tool pair
    assert result[3].role == MessageRole.ASSISTANT
    assert result[3].content is None
    assert result[3].additional_kwargs["tool_calls"][0].tool_name == "weather_api"
    assert result[4].role == MessageRole.TOOL
    assert result[4].additional_kwargs["tool_call_id"] == "weather1"

    # Third assistant+tool pair
    assert result[5].role == MessageRole.ASSISTANT
    assert "Flights are $800-1200" in result[5].content
    assert result[5].additional_kwargs["tool_calls"][0].tool_name == "hotel_search"
    assert result[6].role == MessageRole.TOOL
    assert result[6].additional_kwargs["tool_call_id"] == "hotel1"

    # Fourth assistant+tool pair
    assert result[7].role == MessageRole.ASSISTANT
    assert result[7].content is None
    assert result[7].additional_kwargs["tool_calls"][0].tool_name == "attraction_search"
    assert result[8].role == MessageRole.TOOL
    assert result[8].additional_kwargs["tool_call_id"] == "attraction1"

    # Check that we can extract sources from the history
    _, sources, _ = asyncio.run(process_history_citations(result))
    assert len(sources) == 4

    # Final response
    assert result[9].role == MessageRole.ASSISTANT
    assert "Tokyo trip is planned" in result[9].content

    # Verify all tool calls have unique IDs
    tool_ids = set()
    for msg in result:
        if msg.role == MessageRole.ASSISTANT and "tool_calls" in msg.additional_kwargs:
            tool_id = msg.additional_kwargs["tool_calls"][0].tool_id
            assert tool_id not in tool_ids, f"Duplicate tool ID: {tool_id}"
            tool_ids.add(tool_id)

    assert tool_ids == {"flight1", "weather1", "hotel1", "attraction1"}


def test_tool_results_multiple_in_single_message() -> None:
    # Test when tool results are combined in a single tool message
    messages = [
        MessageInput(role="user", content="Find me a restaurant and check the weather"),
        MessageInput(
            role="assistant",
            content=[
                TextBlock(
                    type="text", text="I'll search for restaurants and check weather"
                ),
                ToolUseBlock(
                    type="tool_use",
                    id="rest1",
                    name="restaurant_search",
                    input={"cuisine": "Italian", "price": "moderate"},
                ),
            ],
        ),
        MessageInput(
            role="assistant",
            content=[
                ToolUseBlock(
                    type="tool_use",
                    id="weather2",
                    name="weather_api",
                    input={"location": "current"},
                )
            ],
        ),
        # Both tool results in a single message
        MessageInput(
            role="assistant",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="rest1",
                    content=[
                        TextBlock(
                            type="text",
                            text="Found: Luigi's Italian (4.5★), Pasta Palace (4.2★)",
                        )
                    ],
                ),
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="weather2",
                    content=[TextBlock(type="text", text="Sunny, 68°F")],
                ),
            ],
        ),
        MessageInput(
            role="assistant",
            content="Great options! Luigi's Italian has 4.5 stars. The weather is sunny and 68°F - perfect for dining out!",
        ),
    ]

    result = MessageInput.convert_from_llama_index_messages(messages)

    # After reordering: user, assistant1, tool1, assistant2, tool2, final assistant
    assert len(result) == 6

    # User message
    assert result[0].role == MessageRole.USER

    # First assistant+tool pair (restaurant search)
    assert result[1].role == MessageRole.ASSISTANT
    assert result[1].additional_kwargs["tool_calls"][0].tool_name == "restaurant_search"
    assert result[2].role == MessageRole.TOOL
    assert result[2].additional_kwargs["tool_call_id"] == "rest1"
    assert "Luigi's Italian" in result[2].content

    # Second assistant+tool pair (weather)
    assert result[3].role == MessageRole.ASSISTANT
    assert result[3].additional_kwargs["tool_calls"][0].tool_name == "weather_api"
    assert result[4].role == MessageRole.TOOL
    assert result[4].additional_kwargs["tool_call_id"] == "weather2"
    assert "Sunny, 68°F" in result[4].content

    # Final assistant response
    assert result[5].role == MessageRole.ASSISTANT
    assert "Luigi's Italian" in result[5].content


def test_assistant_message_only_tools_no_text() -> None:
    """Test assistant message with only tool blocks, no text content."""
    tool_block1 = ToolUseBlock(
        type="tool_use",
        id="tool1",
        name="calculator",
        input={"operation": "add", "x": 1, "y": 2},
    )
    tool_block2 = ToolUseBlock(
        type="tool_use", id="tool2", name="search", input={"query": "weather"}
    )

    message = MessageInput(role="assistant", content=[tool_block1, tool_block2])

    result, tool_uses = message._convert_into_llama_index_messages()

    assert len(result) == 2

    # First message has no text content, just first tool
    assert result[0].role == MessageRole.ASSISTANT
    assert result[0].content is None
    assert len(result[0].additional_kwargs["tool_calls"]) == 1
    assert result[0].additional_kwargs["tool_calls"][0].tool_id == "tool1"

    # Second message has no text content, just second tool
    assert result[1].role == MessageRole.ASSISTANT
    assert result[1].content is None
    assert len(result[1].additional_kwargs["tool_calls"]) == 1
    assert result[1].additional_kwargs["tool_calls"][0].tool_id == "tool2"

    assert len(tool_uses) == 2


def test_mixed_content_blocks_with_tools() -> None:
    """Test assistant message with mixed content: text, tool, result."""
    text_block1 = TextBlock(type="text", text="First text")
    tool_block1 = ToolUseBlock(type="tool_use", id="tool1", name="calc", input={"x": 1})
    tool_result1 = ToolResultBlock(
        type="tool_result",
        tool_use_id="tool1",
        content=[TextBlock(type="text", text="2")],
    )

    message = MessageInput(
        role="assistant", content=[text_block1, tool_block1, tool_result1]
    )

    result, _ = message._convert_into_llama_index_messages()

    assert len(result) == 2

    # First message: first text + first tool
    assert result[0].role == MessageRole.ASSISTANT
    assert result[0].content == "First text"
    assert result[0].additional_kwargs["tool_calls"][0].tool_id == "tool1"

    # Second message: second message is a tool result
    assert result[1].role == MessageRole.TOOL
    assert result[1].additional_kwargs["tool_call_id"] == "tool1"


def test_tool_result_without_prior_tool_use() -> None:
    """Test tool result block when tool_use is not in the current context."""
    tool_result = ToolResultBlock(
        type="tool_result",
        tool_use_id="unknown_tool",
        content=[TextBlock(type="text", text="Some result")],
    )

    message = MessageInput(role="assistant", content=[tool_result])
    result, _ = message._convert_into_llama_index_messages({})

    assert len(result) == 1
    assert result[0].role == MessageRole.TOOL
    assert result[0].additional_kwargs["tool_call_id"] == "unknown_tool"
    assert result[0].additional_kwargs["tool_call_name"] is None  # No prior tool use


def test_empty_tool_use_block() -> None:
    """Test handling of empty/None tool use blocks."""
    text_block = TextBlock(type="text", text="Test")
    # Invalid tool block is rejected by strict pydantic constraints.
    with pytest.raises(ValidationError):
        ToolUseBlock(type="tool_use", id="", name="", input={})

    message = MessageInput(role="assistant", content=[text_block])
    result, _ = message._convert_into_llama_index_messages()

    assert len(result) == 1
    assert result[0].role == MessageRole.ASSISTANT


def test_tool_result_with_multiple_content_blocks() -> None:
    """Test tool result with various content block types."""
    from llama_index.core.schema import NodeWithScore, TextNode

    from private_gpt.events.models import SourceBlock

    text_content = TextBlock(type="text", text="Result text")
    source_content = SourceBlock(
        type="source",
        sources=[
            Chunk.from_node(NodeWithScore(node=TextNode(text="Source 1"))),
            Chunk.from_node(NodeWithScore(node=TextNode(text="Source 2"))),
        ],
    )

    tool_result = ToolResultBlock(
        type="tool_result", tool_use_id="tool1", content=[text_content, source_content]
    )

    message = MessageInput(role="assistant", content=[tool_result])

    # Mock tool_uses
    tool_uses = {
        "tool1": ToolUseBlock(type="tool_use", id="tool1", name="search", input={})
    }

    result, _ = message._convert_into_llama_index_messages(tool_uses)

    assert len(result) == 1
    assert result[0].role == MessageRole.TOOL
    assert result[0].content == "Result text"
    # Should have source blocks in additional_kwargs
    assert "source" in result[0].additional_kwargs


def test_large_number_of_tools() -> None:
    """Test assistant message with many tool blocks."""
    text_block = TextBlock(type="text", text="Using many tools")
    tool_blocks = [
        ToolUseBlock(
            type="tool_use", id=f"tool{i}", name=f"tool_{i}", input={"param": i}
        )
        for i in range(5)
    ]

    message = MessageInput(role="assistant", content=[text_block, *tool_blocks])

    result, tool_uses = message._convert_into_llama_index_messages()

    # Should create 5 separate assistant messages
    assert len(result) == 5

    # First message has text content
    assert result[0].content == "Using many tools"
    assert result[0].additional_kwargs["tool_calls"][0].tool_id == "tool0"

    # Subsequent messages have no text content
    for i in range(1, 5):
        assert result[i].content is None
        assert result[i].additional_kwargs["tool_calls"][0].tool_id == f"tool{i}"

    # All tools should be in tool_uses
    assert len(tool_uses) == 5
    for i in range(5):
        assert f"tool{i}" in tool_uses


def test_convert_from_llama_index_messages_preserves_order() -> None:
    """Test that the overall message order is preserved correctly."""
    messages = [
        MessageInput(role="user", content="Request 1"),
        MessageInput(role="assistant", content="Response 1"),
        MessageInput(role="user", content="Request 2"),
        MessageInput(
            role="assistant",
            content=[
                TextBlock(type="text", text="Using tool"),
                ToolUseBlock(type="tool_use", id="t1", name="tool", input={}),
            ],
        ),
        MessageInput(
            role="assistant",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="t1",
                    content=[TextBlock(type="text", text="Tool result")],
                ),
            ],
        ),
        MessageInput(role="assistant", content="Final response"),
        MessageInput(role="user", content="Request 4"),
        MessageInput(
            role="assistant",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="t2",
                    content=[TextBlock(type="text", text="Tool result")],
                ),
                ToolUseBlock(type="tool_use", id="t2", name="tool", input={}),
                TextBlock(type="text", text="Using tool"),
            ],
        ),
        MessageInput(role="user", content="Request 5"),
        MessageInput(
            role="assistant",
            content=[
                TextBlock(type="text", text="Using tool"),
            ],
        ),
        MessageInput(role="user", content="Request 6"),
    ]

    result = MessageInput.convert_from_llama_index_messages(messages)

    # Verify the sequence: user, assistant, user, assistant, tool, user, assistant
    expected_roles = [
        MessageRole.USER,  # Request 1
        MessageRole.ASSISTANT,  # Response 1
        MessageRole.USER,  # Request 2
        MessageRole.ASSISTANT,  # Using tool
        MessageRole.TOOL,  # Tool result (reordered)
        MessageRole.ASSISTANT,  # Final response
        MessageRole.USER,  # Request 3
        MessageRole.ASSISTANT,  # Final response
        MessageRole.TOOL,
        MessageRole.ASSISTANT,  # Using tool (reordered)
        MessageRole.USER,  # Request 4
        MessageRole.ASSISTANT,  # Using tool (reordered)
        MessageRole.USER,  # Request 5
    ]

    assert len(result) == len(expected_roles)
    for i, expected_role in enumerate(expected_roles):
        assert result[i].role == expected_role, (
            f"Position {i}: expected {expected_role}, got {result[i].role}"
        )


@pytest.mark.parametrize("invalid_role", ["tool", "invalid", "ASSISTANT", "USER"])
def test_invalid_message_roles(invalid_role: str) -> None:
    """Test handling of invalid message roles."""
    with pytest.raises(ValidationError):
        MessageInput(role=invalid_role, content="test")  # type: ignore


def test_image_block_handling() -> None:
    """Test handling of image blocks in content."""
    data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    mime_type = "image/png"

    image_block = ImageBlock.from_base64(data=data, mime_type=mime_type)
    text_block = TextBlock(type="text", text="Check this image")

    message = MessageInput(role="user", content=[text_block, image_block])
    result, _ = message._convert_into_llama_index_messages()

    assert len(result) == 1
    assert result[0].role == MessageRole.USER

    content_blocks = result[0].content
    if isinstance(content_blocks, list):
        text_blocks = [b for b in content_blocks if isinstance(b, LITextBlock)]
        image_blocks = [
            b
            for b in content_blocks
            if isinstance(b, type(content_blocks[0])) and hasattr(b, "image")
        ]

        assert len(text_blocks) >= 1
        assert len(image_blocks) >= 1


def test_convert_multiple_tool_uses_with_interleaved_results() -> None:
    """Test assistant message with multiple tool uses and their results interleaved."""
    messages = [
        MessageInput(role="user", content="Get weather for Madrid and Paris"),
        MessageInput(
            role="assistant",
            content=[
                TextBlock(type="text", text="Using tool1"),
                ToolUseBlock(
                    type="tool_use",
                    id="toolu_madrid",
                    name="get_weather",
                    input={"location": "Madrid"},
                ),
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="toolu_madrid",
                    content=[TextBlock(type="text", text="Madrid: -20°C")],
                    is_error=False,
                ),
                TextBlock(type="text", text="Using tool2"),
                ToolUseBlock(
                    type="tool_use",
                    id="toolu_paris",
                    name="get_weather",
                    input={"location": "Paris"},
                ),
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="toolu_paris",
                    content=[TextBlock(type="text", text="Paris: -20°C")],
                    is_error=False,
                ),
                TextBlock(type="text", text="Weather retrieved for both cities."),
            ],
        ),
        MessageInput(role="user", content="Send email with this info"),
    ]

    result = MessageInput.convert_from_llama_index_messages(messages)

    assert len(result) == 7  # user, assistant, tool1, tool2, user
    assert result[0].role == MessageRole.USER
    assert result[1].role == MessageRole.ASSISTANT
    assert result[2].role == MessageRole.TOOL
    assert result[3].role == MessageRole.ASSISTANT
    assert result[4].role == MessageRole.TOOL
    assert result[5].role == MessageRole.ASSISTANT
    assert result[6].role == MessageRole.USER

    # Verify tool calls are properly connected
    tool_calls = result[1].additional_kwargs["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_id == "toolu_madrid"

    tool_calls = result[3].additional_kwargs["tool_calls"]
    assert tool_calls[0].tool_id == "toolu_paris"

    # Verify tool results maintain proper IDs
    assert result[2].additional_kwargs["tool_call_id"] == "toolu_madrid"
    assert result[4].additional_kwargs["tool_call_id"] == "toolu_paris"

    # Verify text blocks are in the right places
    assert result[1].content == "Using tool1"
    assert result[3].content == "Using tool2"
    assert result[5].content == "Weather retrieved for both cities."


def test_convert_assistant_with_text_and_tool_results_only() -> None:
    """Test assistant message containing only tool results and text (no tool uses)."""
    messages = [
        MessageInput(role="user", content="Previous request"),
        MessageInput(role="assistant", content="I'll help with that"),
        MessageInput(
            role="assistant",
            content=[
                ToolUseBlock(
                    type="tool_use",
                    id="previous_tool_id",
                    name="get_weather",
                    input={"location": "InvalidPlace"},
                ),
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="previous_tool_id",
                    content=[TextBlock(type="text", text="Result data")],
                    is_error=False,
                ),
                TextBlock(type="text", text="Based on the results..."),
            ],
        ),
    ]

    result = MessageInput.convert_from_llama_index_messages(messages)

    assert len(result) == 4
    assert result[0].role == MessageRole.USER
    assert result[1].role == MessageRole.ASSISTANT
    assert result[2].role == MessageRole.TOOL
    assert result[3].role == MessageRole.ASSISTANT


def test_convert_orphaned_tool_results() -> None:
    """Test handling of tool results without corresponding tool uses."""
    messages = [
        MessageInput(role="user", content="Test orphaned results"),
        MessageInput(
            role="assistant",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="missing_tool_id",
                    content=[TextBlock(type="text", text="Orphaned result")],
                    is_error=False,
                ),
                TextBlock(type="text", text="Some response text"),
            ],
        ),
    ]

    result = MessageInput.convert_from_llama_index_messages(messages)

    assert len(result) == 2
    assert len(result[0].additional_kwargs) == 0
    assert len(result[1].additional_kwargs) == 0


def test_convert_multiple_assistant_messages_with_tools() -> None:
    """Test multiple consecutive assistant messages with tool interactions."""
    messages = [
        MessageInput(role="user", content="Multi-step request"),
        MessageInput(
            role="assistant",
            content=[
                ToolUseBlock(
                    type="tool_use", id="step1", name="tool1", input={"param": "value1"}
                )
            ],
        ),
        MessageInput(
            role="assistant",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="step1",
                    content=[TextBlock(type="text", text="Step 1 complete")],
                    is_error=False,
                ),
                ToolUseBlock(
                    type="tool_use", id="step2", name="tool2", input={"param": "value2"}
                ),
            ],
        ),
        MessageInput(
            role="assistant",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="step2",
                    content=[TextBlock(type="text", text="Step 2 complete")],
                    is_error=False,
                ),
                TextBlock(type="text", text="All steps completed"),
            ],
        ),
    ]

    result = MessageInput.convert_from_llama_index_messages(messages)

    assert len(result) == 6  # user, asst, tool, asst, tool, asst
    assert result[0].role == MessageRole.USER
    assert result[1].role == MessageRole.ASSISTANT
    assert result[2].role == MessageRole.TOOL
    assert result[3].role == MessageRole.ASSISTANT
    assert result[4].role == MessageRole.TOOL
    assert result[5].role == MessageRole.ASSISTANT


def test_convert_with_meta_fields() -> None:
    """Test handling of _meta fields in messages."""
    messages = [
        MessageInput(
            role="user",
            content="Test with meta",
        ),
        MessageInput(
            role="assistant",
            content="Response",
        ),
    ]

    result = MessageInput.convert_from_llama_index_messages(messages)

    assert len(result) == 2
    # Verify meta fields are handled appropriately
    assert result[0].role == MessageRole.USER
    assert result[1].role == MessageRole.ASSISTANT


@pytest.mark.parametrize(
    ("message_inputs", "expected_roles_and_tldr"),
    [
        # Case 1: Simple TLDR at end of conversation
        (
            [
                MessageInput(role="user", content="Calculate 2+2"),
                MessageInput(
                    role="assistant",
                    content=[
                        TextBlock(text="Let me calculate"),
                        ToolUseBlock(
                            id="calc1", name="calculator", input={"expr": "2+2"}
                        ),
                        ToolResultBlock(tool_use_id="calc1", content="4"),
                        TLDRBlock(
                            content=[
                                TextBlock(
                                    text="Request calculation for 2+2",
                                    metadata={"type": "tldr", "role": "assistant"},
                                ),
                                TextBlock(
                                    text="4",
                                    metadata={"type": "tldr", "role": "tool"},
                                ),
                            ]
                        ),
                    ],
                ),
            ],
            [
                ("user", False, "Calculate 2+2"),
                ("assistant", True, "Request calculation for 2+2"),
                ("tool", True, "4"),
            ],
        ),
        # Case 2: Multiple TLDR blocks in same assistant message
        (
            [
                MessageInput(role="user", content="Multi-step task"),
                MessageInput(
                    role="assistant",
                    content=[
                        TextBlock(text="Step 1: Addition"),
                        ToolUseBlock(
                            id="add1", name="calculator", input={"expr": "10+5"}
                        ),
                        ToolResultBlock(
                            tool_use_id="add1", content="Calculated 10+5=15"
                        ),
                        TLDRBlock(
                            content=[
                                TextBlock(
                                    text="Request calculation for 10+5",
                                    metadata={"type": "tldr", "role": "assistant"},
                                ),
                                TextBlock(
                                    text="15",
                                    metadata={"type": "tldr", "role": "tool"},
                                ),
                            ]
                        ),
                        TextBlock(text="Step 2: Multiplication"),
                        ToolUseBlock(
                            id="mult2", name="calculator", input={"expr": "15*2"}
                        ),
                        ToolResultBlock(tool_use_id="mult2", content="Calculated 30"),
                        TLDRBlock(
                            content=[
                                TextBlock(
                                    text="Request calculation for 10+5",
                                    metadata={"type": "tldr", "role": "assistant"},
                                ),
                                TextBlock(
                                    text="15",
                                    metadata={"type": "tldr", "role": "tool"},
                                ),
                                TextBlock(
                                    text="Request calculation for 15*2",
                                    metadata={"type": "tldr", "role": "assistant"},
                                ),
                                TextBlock(
                                    text="30",
                                    metadata={"type": "tldr", "role": "tool"},
                                ),
                            ]
                        ),
                    ],
                ),
            ],
            [
                ("user", False, "Multi-step task"),
                ("assistant", True, "Request calculation for 10+5"),
                ("tool", True, "15"),
                ("assistant", True, "Request calculation for 15*2"),
                ("tool", True, "30"),
            ],
        ),
        # Case 3: TLDR followed by another user query
        (
            [
                MessageInput(role="user", content="First calculation"),
                MessageInput(
                    role="assistant",
                    content=[
                        TextBlock(text="Calculating..."),
                        ToolUseBlock(
                            id="calc3", name="calculator", input={"expr": "7*8"}
                        ),
                        ToolResultBlock(tool_use_id="calc3", content="56"),
                        TLDRBlock(
                            content=[
                                TextBlock(
                                    text="Request calculation for 7*8",
                                    metadata={"type": "tldr", "role": "assistant"},
                                ),
                                TextBlock(
                                    text="56",
                                    metadata={"type": "tldr", "role": "tool"},
                                ),
                            ]
                        ),
                        TextBlock(text="The result is 56"),
                    ],
                ),
                MessageInput(role="user", content="What about division?"),
            ],
            [
                ("user", False, "First calculation"),
                ("assistant", True, "Request calculation for 7*8"),
                ("tool", True, "56"),
                ("assistant", False, "The result is 56"),
                ("user", False, "What about division?"),
            ],
        ),
        # Case 4: TLDR followed by an user query
        (
            [
                MessageInput(role="user", content="This message has to be dropped"),
                MessageInput(
                    role="assistant", content="This message has to be dropped"
                ),
                MessageInput(role="user", content="This message has to be dropped"),
                MessageInput(
                    role="assistant", content="This message has to be dropped"
                ),
                MessageInput(role="user", content="First calculation"),
                MessageInput(
                    role="assistant",
                    content=[
                        TLDRBlock(
                            content=[
                                TextBlock(
                                    text="Drop 1",
                                    metadata={"type": "tldr", "role": "user"},
                                ),
                                TextBlock(
                                    text="Drop 2",
                                    metadata={"type": "tldr", "role": "assistant"},
                                ),
                                TextBlock(
                                    text="Drop 3",
                                    metadata={"type": "tldr", "role": "user"},
                                ),
                                TextBlock(
                                    text="Drop 4",
                                    metadata={"type": "tldr", "role": "assistant"},
                                ),
                            ]
                        ),
                        TextBlock(text="Calculating..."),
                        ToolUseBlock(
                            id="calc3", name="calculator", input={"expr": "7*8"}
                        ),
                        ToolResultBlock(tool_use_id="calc3", content="56"),
                        TextBlock(text="The result is 56"),
                    ],
                ),
                MessageInput(role="user", content="What about division?"),
            ],
            [
                ("user", True, "Drop 1"),
                ("assistant", True, "Drop 2"),
                ("user", True, "Drop 3"),
                ("assistant", True, "Drop 4"),
                ("user", False, "First calculation"),
                ("assistant", False, "Calculating..."),
                ("tool", False, "56"),
                ("assistant", False, "The result is 56"),
                ("user", False, "What about division?"),
            ],
        ),
        # Case 5: TLDR in left and right
        (
            [
                MessageInput(role="user", content="This message has to be dropped"),
                MessageInput(
                    role="assistant", content="This message has to be dropped"
                ),
                MessageInput(role="user", content="This message has to be dropped"),
                MessageInput(
                    role="assistant", content="This message has to be dropped"
                ),
                MessageInput(role="user", content="First calculation"),
                MessageInput(
                    role="assistant",
                    content=[
                        TLDRBlock(
                            content=[
                                TextBlock(
                                    text="Drop 1",
                                    metadata={"type": "tldr", "role": "user"},
                                ),
                                TextBlock(
                                    text="Drop 2",
                                    metadata={"type": "tldr", "role": "assistant"},
                                ),
                                TextBlock(
                                    text="Drop 3",
                                    metadata={"type": "tldr", "role": "user"},
                                ),
                                TextBlock(
                                    text="Drop 4",
                                    metadata={"type": "tldr", "role": "assistant"},
                                ),
                            ]
                        ),
                        TextBlock(text="Calculating..."),
                        ToolUseBlock(
                            id="calc3", name="calculator", input={"expr": "7*8"}
                        ),
                        ToolResultBlock(tool_use_id="calc3", content="56"),
                        TLDRBlock(
                            content=[
                                TextBlock(
                                    text="Request calculation for 7*8",
                                    metadata={"type": "tldr", "role": "assistant"},
                                ),
                                TextBlock(
                                    text="56",
                                    metadata={"type": "tldr", "role": "tool"},
                                ),
                            ]
                        ),
                        TextBlock(text="The result is 56"),
                    ],
                ),
                MessageInput(role="user", content="What about division?"),
            ],
            [
                ("user", True, "Drop 1"),
                ("assistant", True, "Drop 2"),
                ("user", True, "Drop 3"),
                ("assistant", True, "Drop 4"),
                ("user", False, "First calculation"),
                ("assistant", True, "Request calculation for 7*8"),
                ("tool", True, "56"),
                ("assistant", False, "The result is 56"),
                ("user", False, "What about division?"),
            ],
        ),
        # Case 6: Several TLDR in left and right (even by level)
        (
            [
                MessageInput(role="user", content="This message has to be dropped"),
                MessageInput(
                    role="assistant", content="This message has to be dropped"
                ),
                MessageInput(role="user", content="This message has to be dropped"),
                MessageInput(
                    role="assistant", content="This message has to be dropped"
                ),
                MessageInput(role="user", content="First calculation"),
                MessageInput(
                    role="assistant",
                    content=[
                        TLDRBlock(
                            content=[
                                TextBlock(
                                    text="Drop 1",
                                    metadata={"type": "tldr", "role": "user"},
                                ),
                                TextBlock(
                                    text="Drop 2",
                                    metadata={"type": "tldr", "role": "assistant"},
                                ),
                                TextBlock(
                                    text="Drop 3",
                                    metadata={"type": "tldr", "role": "user"},
                                ),
                                TextBlock(
                                    text="Drop 4",
                                    metadata={"type": "tldr", "role": "assistant"},
                                ),
                            ]
                        ),
                        TextBlock(text="Calculating..."),
                        ToolUseBlock(
                            id="calc3", name="calculator", input={"expr": "7*8"}
                        ),
                        ToolResultBlock(tool_use_id="calc3", content="56"),
                        TLDRBlock(
                            content=[
                                TextBlock(
                                    text="Request calculation for 7*8",
                                    metadata={"type": "tldr", "role": "assistant"},
                                ),
                                TextBlock(
                                    text="56",
                                    metadata={"type": "tldr", "role": "tool"},
                                ),
                            ]
                        ),
                        TextBlock(text="The result is 56"),
                    ],
                ),
                MessageInput(role="user", content="What about division?"),
                MessageInput(
                    role="assistant",
                    content=[
                        TLDRBlock(
                            content=[
                                TextBlock(
                                    text="Drop 1",
                                    metadata={"type": "tldr", "role": "user"},
                                ),
                                TextBlock(
                                    text="Drop 2",
                                    metadata={"type": "tldr", "role": "assistant"},
                                ),
                                TextBlock(
                                    text="Drop 3",
                                    metadata={"type": "tldr", "role": "user"},
                                ),
                                TextBlock(
                                    text="Drop 4",
                                    metadata={"type": "tldr", "role": "assistant"},
                                ),
                                TextBlock(
                                    text="What about division? Dropped",
                                    metadata={"type": "tldr", "role": "user"},
                                ),
                                TextBlock(
                                    text="The result is 56 Dropped",
                                    metadata={"type": "tldr", "role": "assistant"},
                                ),
                            ]
                        ),
                        TextBlock(text="Calculating..."),
                        ToolUseBlock(
                            id="calc3", name="calculator", input={"expr": "7*8"}
                        ),
                        ToolResultBlock(tool_use_id="calc3", content="56"),
                        TLDRBlock(
                            content=[
                                TextBlock(
                                    text="Request calculation for 7*8",
                                    metadata={"type": "tldr", "role": "assistant"},
                                ),
                                TextBlock(
                                    text="56",
                                    metadata={"type": "tldr", "role": "tool"},
                                ),
                            ]
                        ),
                        TextBlock(text="The result is 56"),
                    ],
                ),
            ],
            [
                ("user", True, "Drop 1"),
                ("assistant", True, "Drop 2"),
                ("user", True, "Drop 3"),
                ("assistant", True, "Drop 4"),
                ("user", True, "What about division? Dropped"),
                ("assistant", True, "The result is 56 Dropped"),
                ("user", False, "What about division?"),
                ("assistant", True, "Request calculation for 7*8"),
                ("tool", True, "56"),
                ("assistant", False, "The result is 56"),
            ],
        ),
        # Case 4: No TLDR blocks
        (
            [
                MessageInput(role="user", content="Simple query"),
                MessageInput(
                    role="assistant",
                    content=[
                        TextBlock(text="Simple response"),
                        ToolUseBlock(
                            id="tool1", name="helper", input={"param": "value"}
                        ),
                        ToolResultBlock(tool_use_id="tool1", content="Tool output"),
                        TextBlock(text="Second simple response"),
                    ],
                ),
                MessageInput(role="user", content="Follow up"),
            ],
            [
                ("user", False, "Simple query"),
                ("assistant", False, "Simple response"),
                ("tool", False, "Tool output"),
                ("assistant", False, "Second simple response"),
                ("user", False, "Follow up"),
            ],
        ),
        # Case 5: Empty TLDR block
        (
            [
                MessageInput(role="user", content="Test empty TLDR"),
                MessageInput(
                    role="assistant",
                    content=[
                        TextBlock(text="Some response"),
                        TLDRBlock(content=[]),
                    ],
                ),
            ],
            [
                ("user", False, "Test empty TLDR"),
                ("assistant", False, "Some response"),
            ],
        ),
        # Case 7: Explicit right-side TLDR only
        (
            [
                MessageInput(role="user", content="Calculate 5+5"),
                MessageInput(
                    role="assistant",
                    content=[
                        TextBlock(text="Computing..."),
                        ToolUseBlock(
                            id="calc_right", name="calculator", input={"expr": "5+5"}
                        ),
                        ToolResultBlock(tool_use_id="calc_right", content="10"),
                        TLDRBlock(
                            content=[
                                TextBlock(
                                    text="Requested 5+5",
                                    metadata={
                                        "type": "tldr",
                                        "role": "assistant",
                                        "tldr_side": "right",
                                    },
                                ),
                                TextBlock(
                                    text="Result: 10",
                                    metadata={
                                        "type": "tldr",
                                        "role": "tool",
                                        "tldr_side": "right",
                                    },
                                ),
                            ],
                            tldr_side="right",
                        ),
                        TextBlock(text="Done!"),
                    ],
                ),
            ],
            [
                ("user", False, "Calculate 5+5"),
                ("assistant", True, "Requested 5+5"),
                ("tool", True, "Result: 10"),
                ("assistant", False, "Done!"),
            ],
        ),
        # Case 8: Explicit left-side TLDR only
        (
            [
                MessageInput(role="user", content="Old message 1"),
                MessageInput(role="assistant", content="Old response 1"),
                MessageInput(role="user", content="Old message 2"),
                MessageInput(role="assistant", content="Old response 2"),
                MessageInput(role="user", content="Current query"),
                MessageInput(
                    role="assistant",
                    content=[
                        TLDRBlock(
                            content=[
                                TextBlock(
                                    text="Summary: Old message 1",
                                    metadata={
                                        "type": "tldr",
                                        "role": "user",
                                        "tldr_side": "left",
                                    },
                                ),
                                TextBlock(
                                    text="Summary: Old response 1",
                                    metadata={
                                        "type": "tldr",
                                        "role": "assistant",
                                        "tldr_side": "left",
                                    },
                                ),
                                TextBlock(
                                    text="Summary: Old message 2",
                                    metadata={
                                        "type": "tldr",
                                        "role": "user",
                                        "tldr_side": "left",
                                    },
                                ),
                                TextBlock(
                                    text="Summary: Old response 2",
                                    metadata={
                                        "type": "tldr",
                                        "role": "assistant",
                                        "tldr_side": "left",
                                    },
                                ),
                            ],
                            tldr_side="left",
                        ),
                        TextBlock(text="Current response"),
                    ],
                ),
            ],
            [
                ("user", True, "Summary: Old message 1"),
                ("assistant", True, "Summary: Old response 1"),
                ("user", True, "Summary: Old message 2"),
                ("assistant", True, "Summary: Old response 2"),
                ("user", False, "Current query"),
                ("assistant", False, "Current response"),
            ],
        ),
        # Case 9: Explicit combination of left and right TLDR
        (
            [
                MessageInput(role="user", content="Historic query 1"),
                MessageInput(role="assistant", content="Historic response 1"),
                MessageInput(role="user", content="Historic query 2"),
                MessageInput(role="assistant", content="Historic response 2"),
                MessageInput(role="user", content="Recent query"),
                MessageInput(
                    role="assistant",
                    content=[
                        TLDRBlock(
                            content=[
                                TextBlock(
                                    text="Left summary 1",
                                    metadata={
                                        "type": "tldr",
                                        "role": "user",
                                        "tldr_side": "left",
                                    },
                                ),
                                TextBlock(
                                    text="Left summary 2",
                                    metadata={
                                        "type": "tldr",
                                        "role": "assistant",
                                        "tldr_side": "left",
                                    },
                                ),
                            ],
                            tldr_side="left",
                        ),
                        TextBlock(text="Processing..."),
                        ToolUseBlock(
                            id="tool_combo", name="processor", input={"data": "test"}
                        ),
                        ToolResultBlock(tool_use_id="tool_combo", content="Processed"),
                        TLDRBlock(
                            content=[
                                TextBlock(
                                    text="Right summary 1",
                                    metadata={
                                        "type": "tldr",
                                        "role": "assistant",
                                        "tldr_side": "right",
                                    },
                                ),
                                TextBlock(
                                    text="Right summary 2",
                                    metadata={
                                        "type": "tldr",
                                        "role": "tool",
                                        "tldr_side": "right",
                                    },
                                ),
                            ],
                            tldr_side="right",
                        ),
                        TextBlock(text="Complete"),
                    ],
                ),
            ],
            [
                ("user", True, "Left summary 1"),
                ("assistant", True, "Left summary 2"),
                ("user", False, "Recent query"),
                ("assistant", True, "Right summary 1"),
                ("tool", True, "Right summary 2"),
                ("assistant", False, "Complete"),
            ],
        ),
        # Case 10: Multiple right-side TLDRs in sequence
        (
            [
                MessageInput(role="user", content="Do multiple tasks"),
                MessageInput(
                    role="assistant",
                    content=[
                        TextBlock(text="Task 1"),
                        ToolUseBlock(id="t1", name="task1", input={}),
                        ToolResultBlock(tool_use_id="t1", content="T1 done"),
                        TLDRBlock(
                            content=[
                                TextBlock(
                                    text="Task 1 summarize",
                                    metadata={
                                        "type": "tldr",
                                        "role": "assistant",
                                        "tldr_side": "right",
                                    },
                                ),
                                TextBlock(
                                    text="Task 1 summarized",
                                    metadata={
                                        "type": "tldr",
                                        "role": "tool",
                                        "tldr_side": "right",
                                    },
                                ),
                            ],
                            tldr_side="right",
                        ),
                        TextBlock(text="Task 2"),
                        ToolUseBlock(id="t2", name="task2", input={}),
                        ToolResultBlock(tool_use_id="t2", content="T2 done"),
                        TLDRBlock(
                            content=[
                                TextBlock(
                                    text="Task 2 summarize",
                                    metadata={
                                        "type": "tldr",
                                        "role": "assistant",
                                        "tldr_side": "right",
                                    },
                                ),
                                TextBlock(
                                    text="Task 2 summarized",
                                    metadata={
                                        "type": "tldr",
                                        "role": "tool",
                                        "tldr_side": "right",
                                    },
                                ),
                            ],
                            tldr_side="right",
                        ),
                        TextBlock(text="Complete"),
                    ],
                ),
            ],
            [
                ("user", False, "Do multiple tasks"),
                ("assistant", True, "Task 1 summarize"),
                ("tool", True, "Task 1 summarized"),
                ("assistant", True, "Task 2 summarize"),
                ("tool", True, "Task 2 summarized"),
                ("assistant", False, "Complete"),
            ],
        ),
        # Case 11: Double right TLDR with prior conversational history
        (
            [
                MessageInput(role="user", content="Old query"),
                MessageInput(role="assistant", content=[TextBlock(text="Old answer")]),
                MessageInput(role="user", content="Current query"),
                MessageInput(
                    role="assistant",
                    content=[
                        TextBlock(text="Phase 1"),
                        ToolUseBlock(
                            id="dr1", name="calculator", input={"expr": "3+4"}
                        ),
                        ToolResultBlock(tool_use_id="dr1", content="7"),
                        TLDRBlock(
                            tldr_side="right",
                            content=[
                                TextBlock(
                                    text="Do 3+4",
                                    metadata={
                                        "type": "tldr",
                                        "role": "assistant",
                                        "tldr_side": "right",
                                    },
                                ),
                                TextBlock(
                                    text="7",
                                    metadata={
                                        "type": "tldr",
                                        "role": "tool",
                                        "tldr_side": "right",
                                    },
                                ),
                            ],
                        ),
                        TextBlock(text="Phase 2"),
                        ToolUseBlock(
                            id="dr2", name="calculator", input={"expr": "7*3"}
                        ),
                        ToolResultBlock(tool_use_id="dr2", content="21"),
                        TLDRBlock(
                            tldr_side="right",
                            content=[
                                TextBlock(
                                    text="Do 3+4",
                                    metadata={
                                        "type": "tldr",
                                        "role": "assistant",
                                        "tldr_side": "right",
                                    },
                                ),
                                TextBlock(
                                    text="7",
                                    metadata={
                                        "type": "tldr",
                                        "role": "tool",
                                        "tldr_side": "right",
                                    },
                                ),
                                TextBlock(
                                    text="Do 7*3",
                                    metadata={
                                        "type": "tldr",
                                        "role": "assistant",
                                        "tldr_side": "right",
                                    },
                                ),
                                TextBlock(
                                    text="21",
                                    metadata={
                                        "type": "tldr",
                                        "role": "tool",
                                        "tldr_side": "right",
                                    },
                                ),
                            ],
                        ),
                        TextBlock(text="Final response"),
                    ],
                ),
            ],
            [
                ("user", False, "Old query"),
                ("assistant", False, "Old answer"),
                ("user", False, "Current query"),
                ("assistant", True, "Do 3+4"),
                ("tool", True, "7"),
                ("assistant", True, "Do 7*3"),
                ("tool", True, "21"),
                ("assistant", False, "Final response"),
            ],
        ),
        # Case 12: Double right TLDR after left TLDR in same assistant message
        (
            [
                MessageInput(role="user", content="Legacy 1"),
                MessageInput(role="assistant", content="Legacy A"),
                MessageInput(role="user", content="Legacy 2"),
                MessageInput(role="assistant", content="Legacy B"),
                MessageInput(role="user", content="Active task"),
                MessageInput(
                    role="assistant",
                    content=[
                        TLDRBlock(
                            tldr_side="left",
                            content=[
                                TextBlock(
                                    text="L-user-1",
                                    metadata={
                                        "type": "tldr",
                                        "role": "user",
                                        "tldr_side": "left",
                                    },
                                ),
                                TextBlock(
                                    text="L-asst-1",
                                    metadata={
                                        "type": "tldr",
                                        "role": "assistant",
                                        "tldr_side": "left",
                                    },
                                ),
                            ],
                        ),
                        TextBlock(text="Step one"),
                        ToolUseBlock(id="mix1", name="helper", input={}),
                        ToolResultBlock(tool_use_id="mix1", content="ok1"),
                        TLDRBlock(
                            tldr_side="right",
                            content=[
                                TextBlock(
                                    text="R-step-1",
                                    metadata={
                                        "type": "tldr",
                                        "role": "assistant",
                                        "tldr_side": "right",
                                    },
                                ),
                                TextBlock(
                                    text="R-ok-1",
                                    metadata={
                                        "type": "tldr",
                                        "role": "tool",
                                        "tldr_side": "right",
                                    },
                                ),
                            ],
                        ),
                        TextBlock(text="Step two"),
                        ToolUseBlock(id="mix2", name="helper", input={}),
                        ToolResultBlock(tool_use_id="mix2", content="ok2"),
                        TLDRBlock(
                            tldr_side="right",
                            content=[
                                TextBlock(
                                    text="R-step-1",
                                    metadata={
                                        "type": "tldr",
                                        "role": "assistant",
                                        "tldr_side": "right",
                                    },
                                ),
                                TextBlock(
                                    text="R-ok-1",
                                    metadata={
                                        "type": "tldr",
                                        "role": "tool",
                                        "tldr_side": "right",
                                    },
                                ),
                                TextBlock(
                                    text="R-step-2",
                                    metadata={
                                        "type": "tldr",
                                        "role": "assistant",
                                        "tldr_side": "right",
                                    },
                                ),
                                TextBlock(
                                    text="R-ok-2",
                                    metadata={
                                        "type": "tldr",
                                        "role": "tool",
                                        "tldr_side": "right",
                                    },
                                ),
                            ],
                        ),
                        TextBlock(text="Done active task"),
                    ],
                ),
            ],
            [
                ("user", True, "L-user-1"),
                ("assistant", True, "L-asst-1"),
                ("user", False, "Active task"),
                ("assistant", True, "R-step-1"),
                ("tool", True, "R-ok-1"),
                ("assistant", True, "R-step-2"),
                ("tool", True, "R-ok-2"),
                ("assistant", False, "Done active task"),
            ],
        ),
        # Case 13: Double right TLDR with previous user block still preserved
        (
            [
                MessageInput(role="user", content="First user"),
                MessageInput(
                    role="assistant",
                    content=[
                        TextBlock(text="First answer"),
                    ],
                ),
                MessageInput(role="user", content="Second user"),
                MessageInput(
                    role="assistant",
                    content=[
                        TextBlock(text="Run alpha"),
                        ToolUseBlock(id="p1", name="pipe", input={}),
                        ToolResultBlock(tool_use_id="p1", content="alpha"),
                        TLDRBlock(
                            tldr_side="right",
                            content=[
                                TextBlock(
                                    text="alpha step",
                                    metadata={
                                        "type": "tldr",
                                        "role": "assistant",
                                        "tldr_side": "right",
                                    },
                                ),
                                TextBlock(
                                    text="alpha out",
                                    metadata={
                                        "type": "tldr",
                                        "role": "tool",
                                        "tldr_side": "right",
                                    },
                                ),
                            ],
                        ),
                        TextBlock(text="Run beta"),
                        ToolUseBlock(id="p2", name="pipe", input={}),
                        ToolResultBlock(tool_use_id="p2", content="beta"),
                        TLDRBlock(
                            tldr_side="right",
                            content=[
                                TextBlock(
                                    text="alpha step",
                                    metadata={
                                        "type": "tldr",
                                        "role": "assistant",
                                        "tldr_side": "right",
                                    },
                                ),
                                TextBlock(
                                    text="alpha out",
                                    metadata={
                                        "type": "tldr",
                                        "role": "tool",
                                        "tldr_side": "right",
                                    },
                                ),
                                TextBlock(
                                    text="beta step",
                                    metadata={
                                        "type": "tldr",
                                        "role": "assistant",
                                        "tldr_side": "right",
                                    },
                                ),
                                TextBlock(
                                    text="beta out",
                                    metadata={
                                        "type": "tldr",
                                        "role": "tool",
                                        "tldr_side": "right",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
            [
                ("user", False, "First user"),
                ("assistant", False, "First answer"),
                ("user", False, "Second user"),
                ("assistant", True, "alpha step"),
                ("tool", True, "alpha out"),
                ("assistant", True, "beta step"),
                ("tool", True, "beta out"),
            ],
        ),
    ],
)
def test_reorder_tldr_messages_with_message_input(
    message_inputs: list[MessageInput],
    expected_roles_and_tldr: list[tuple[str, bool, str]],
) -> None:
    result = MessageInput.convert_from_llama_index_messages(message_inputs)

    assert len(result) == len(expected_roles_and_tldr), (
        f"Expected {len(expected_roles_and_tldr)} messages, got {len(result)}"
    )

    for i, (actual_msg, (expected_role, has_tldr, expected_content)) in enumerate(
        zip(result, expected_roles_and_tldr, strict=False)
    ):
        assert actual_msg.role.value == expected_role, (
            f"Message {i}: expected role {expected_role}, got {actual_msg.role.value}"
        )
        assert bool("tldr" in actual_msg.additional_kwargs) == has_tldr, (
            f"Message {i}: TLDR flag mismatch"
        )
        assert actual_msg.content == expected_content, (
            f"Message {i}: expected content '{expected_content}', got '{actual_msg.content}'"
        )


def test_bug_three_consecutive_same_role_with_tldr_pattern() -> None:
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="User message 1",
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Regular message 1",
            additional_kwargs={},  # No TLDR
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="TLDR message 2",
            additional_kwargs={"tldr": True},
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="TLDR message 3",
            additional_kwargs={"tldr": True},
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="TLDR message 4",
            additional_kwargs={"tldr": True},
        ),
    ]

    result = MessageInput._reorder_tldr_messages(messages)
    assert len(result) > 0


def test_accumulated_right_tldr_with_thinking_blocks_causes_consecutive_assistant_messages() -> (
    None
):
    """Regression: exact structure from production crash.

    thinking → t1 → result1
    thinking → t2 → result2
    TLDR1: [ASST(t1), TOOL(-), ASST(t2), TOOL(-)]
    thinking → t3 → result3
    TLDR2: [ASST(t1), TOOL(-), ASST(t2), TOOL(-), ASST(t3), TOOL(-)]
    thinking → t4 → result4  (no TLDR)

    TOOL('-') entries are identical ChatMessages → add_result deduplicates them.
    TLDR2 skips ASST(t1), TOOL(-), ASST(t2) as already added, skips TOOL(-),
    adds ASST(t3), then skips TOOL(-) → ASST(t2) followed by ASST(t3) directly
    → consecutive ASSISTANT → ValueError in _validate_message_order.
    """
    messages = [
        MessageInput(role="user", content="query"),
        MessageInput(
            role="assistant",
            content=[
                ThinkingBlock(
                    type="thinking", thinking="thinking 1", signature="sig_1"
                ),
                ToolUseBlock(type="tool_use", id="t1", name="tool1", input={}),
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="t1",
                    content=[TextBlock(type="text", text="result1")],
                ),
                ThinkingBlock(
                    type="thinking", thinking="thinking 2", signature="sig_2"
                ),
                ToolUseBlock(type="tool_use", id="t2", name="tool2", input={}),
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="t2",
                    content=[TextBlock(type="text", text="result2")],
                ),
                TLDRBlock(
                    tldr_side="right",
                    content=[
                        TextBlock(
                            text="[t1]", metadata={"type": "tldr", "role": "assistant"}
                        ),
                        TextBlock(text="-", metadata={"type": "tldr", "role": "tool"}),
                        TextBlock(
                            text="[t2]", metadata={"type": "tldr", "role": "assistant"}
                        ),
                        TextBlock(text="-", metadata={"type": "tldr", "role": "tool"}),
                    ],
                ),
                ThinkingBlock(
                    type="thinking", thinking="thinking 3", signature="sig_3"
                ),
                ToolUseBlock(type="tool_use", id="t3", name="tool3", input={}),
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="t3",
                    content=[TextBlock(type="text", text="result3")],
                ),
                TLDRBlock(
                    tldr_side="right",
                    content=[
                        TextBlock(
                            text="[t1]", metadata={"type": "tldr", "role": "assistant"}
                        ),
                        TextBlock(text="-", metadata={"type": "tldr", "role": "tool"}),
                        TextBlock(
                            text="[t2]", metadata={"type": "tldr", "role": "assistant"}
                        ),
                        TextBlock(text="-", metadata={"type": "tldr", "role": "tool"}),
                        TextBlock(
                            text="[t3]", metadata={"type": "tldr", "role": "assistant"}
                        ),
                        TextBlock(text="-", metadata={"type": "tldr", "role": "tool"}),
                    ],
                ),
                ThinkingBlock(
                    type="thinking", thinking="thinking 4", signature="sig_4"
                ),
                ToolUseBlock(type="tool_use", id="t4", name="tool4", input={}),
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="t4",
                    content=[TextBlock(type="text", text="result4")],
                ),
            ],
        ),
    ]

    result = MessageInput.convert_from_llama_index_messages(messages)

    roles = [m.role for m in result]
    for i in range(len(roles) - 1):
        assert roles[i] != roles[i + 1], (
            f"Consecutive same-role at positions {i}/{i + 1}: {roles}"
        )
