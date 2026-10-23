import asyncio

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.chat.processors.chat_history.memory.tldr_processor import (
    CondenseResponse,
)
from private_gpt.components.chat.processors.chat_history.memory.tldr_utils import (
    trim_to_last_tldr,
)
from private_gpt.components.chat.processors.chat_history.memory.utils.condenser import (
    build_condensed_history,
)
from private_gpt.events.models import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    TextBlock,
    TLDRBlock,
    TLDRDelta,
)
from private_gpt.server.chat.interceptors.condensation_interceptor import (
    _SENTINEL,
    _condensation_producer,
)


class TestTLDRBlockModel:
    """Test TLDRBlock model with tldr_side field."""

    def test_tldr_block_default_side(self):
        """Test that TLDRBlock defaults to left side."""
        block = TLDRBlock(content=[TextBlock(text="Summary")])
        assert block.tldr_side == "left"

    def test_tldr_block_explicit_left(self):
        """Test TLDRBlock with explicit left side."""
        block = TLDRBlock(content=[TextBlock(text="Summary")], tldr_side="left")
        assert block.tldr_side == "left"

    def test_tldr_block_explicit_right(self):
        """Test TLDRBlock with explicit right side."""
        block = TLDRBlock(content=[TextBlock(text="Summary")], tldr_side="right")
        assert block.tldr_side == "right"

    def test_tldr_block_serialization(self):
        """Test TLDRBlock serialization includes tldr_side."""
        block = TLDRBlock(content=[TextBlock(text="Summary")], tldr_side="right")
        data = block.model_dump()
        assert "tldr_side" in data
        assert data["tldr_side"] == "right"

    def test_tldr_block_deserialization(self):
        """Test TLDRBlock deserialization from dict."""
        data = {
            "type": "tldr",
            "content": [{"type": "text", "text": "Summary"}],
            "tldr_side": "right",
        }
        block = TLDRBlock(**data)
        assert block.tldr_side == "right"
        assert len(block.content) == 1


class TestTrimToLastTLDR:
    """Test trim_to_last_tldr function with both left and right TLDR."""

    def test_empty_history(self):
        """Test with empty chat history."""
        result = trim_to_last_tldr([])
        assert result == []

    def test_no_tldr_messages(self):
        """Test with no TLDR messages."""
        history = [
            ChatMessage(role=MessageRole.USER, content="Hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi there"),
        ]
        result = trim_to_last_tldr(history)
        assert len(result) == 2
        assert result == history

    def test_no_user_message(self):
        """Test when there's no user message."""
        history = [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Summary",
                additional_kwargs={"tldr": "left"},
            ),
        ]
        result = trim_to_last_tldr(history)
        assert result == history

    def test_left_tldr_before_user(self):
        """Test left TLDR before last user message."""
        history = [
            ChatMessage(role=MessageRole.USER, content="First"),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Old summary",
                additional_kwargs={"tldr": "left"},
            ),
            ChatMessage(role=MessageRole.USER, content="Second"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Response"),
        ]
        result = trim_to_last_tldr(history)
        # Should keep from the TLDR onwards
        assert len(result) == 3
        assert result[0].additional_kwargs.get("tldr") == "left"

    def test_right_tldr_after_user(self):
        """Test right TLDR after last user message."""
        history = [
            ChatMessage(role=MessageRole.USER, content="Query"),
            ChatMessage(
                role=MessageRole.TOOL,
                content="Tool summary",
                additional_kwargs={"tldr": "right"},
            ),
            ChatMessage(role=MessageRole.ASSISTANT, content="Final response"),
        ]
        result = trim_to_last_tldr(history)
        # Should keep from the TLDR onwards after user
        assert len(result) == 3
        assert result[1].additional_kwargs.get("tldr") == "right"

    def test_both_left_and_right_tldr(self):
        """Test with both left and right TLDR messages."""
        history = [
            ChatMessage(role=MessageRole.USER, content="Old msg"),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Left summary",
                additional_kwargs={"tldr": "left"},
            ),
            ChatMessage(role=MessageRole.USER, content="Current query"),
            ChatMessage(
                role=MessageRole.TOOL,
                content="Right summary",
                additional_kwargs={"tldr": "right"},
            ),
            ChatMessage(role=MessageRole.ASSISTANT, content="Response"),
        ]
        result = trim_to_last_tldr(history)
        assert len(result) == 4
        # Should have left TLDR, user, right TLDR, and assistant
        assert result[0].additional_kwargs.get("tldr") == "left"
        assert result[1].role == MessageRole.USER
        assert result[2].additional_kwargs.get("tldr") == "right"

    def test_multiple_left_tldrs(self):
        """Test with multiple left TLDR messages."""
        history = [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="First summary",
                additional_kwargs={"tldr": "left"},
            ),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Second summary",
                additional_kwargs={"tldr": "left"},
            ),
            ChatMessage(role=MessageRole.USER, content="Query"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Response"),
        ]
        result = trim_to_last_tldr(history)
        # Should keep from the first of consecutive TLDRs
        assert len(result) == 4
        assert result[0].content == "First summary"
        assert result[1].content == "Second summary"

    def test_multiple_right_tldrs(self):
        """Test with multiple right TLDR messages."""
        history = [
            ChatMessage(role=MessageRole.USER, content="Query"),
            ChatMessage(
                role=MessageRole.TOOL,
                content="First tool summary",
                additional_kwargs={"tldr": "right"},
            ),
            ChatMessage(
                role=MessageRole.TOOL,
                content="Second tool summary",
                additional_kwargs={"tldr": "right"},
            ),
            ChatMessage(role=MessageRole.ASSISTANT, content="Response"),
        ]
        result = trim_to_last_tldr(history)
        # Should keep from FIRST right TLDR onwards (to preserve all new messages)
        assert len(result) == 4
        assert result[1].content == "First tool summary"
        assert result[2].content == "Second tool summary"
        assert result[3].content == "Response"

    def test_right_tldr_preserves_all_after_messages(self):
        """Test that right TLDR keeps ALL messages after it, not discarding any."""
        history = [
            ChatMessage(role=MessageRole.USER, content="Query"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Before TLDR 1"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Before TLDR 2"),
            ChatMessage(
                role=MessageRole.TOOL,
                content="Tool summary",
                additional_kwargs={"tldr": "right"},
            ),
            ChatMessage(role=MessageRole.ASSISTANT, content="After TLDR 1"),
            ChatMessage(role=MessageRole.ASSISTANT, content="After TLDR 2"),
            ChatMessage(role=MessageRole.ASSISTANT, content="After TLDR 3"),
        ]
        result = trim_to_last_tldr(history)
        # Should discard messages BEFORE TLDR, but keep TLDR and ALL after it
        assert len(result) == 5  # User + TLDR + 3 after messages
        assert result[0].role == MessageRole.USER
        assert result[1].additional_kwargs.get("tldr") == "right"
        assert result[2].content == "After TLDR 1"
        assert result[3].content == "After TLDR 2"
        assert result[4].content == "After TLDR 3"


class TestBuildCondensedHistory:
    """Test build_condensed_history function."""

    def test_empty_conversation(self):
        """Test with empty conversation history."""
        system_msgs = [ChatMessage(role=MessageRole.SYSTEM, content="System prompt")]
        conv_history = []

        with pytest.raises(ValueError, match="No user message found"):
            build_condensed_history(system_msgs, conv_history)

    def test_no_tldr_messages(self):
        """Test with no TLDR messages."""
        system_msgs = []
        conv_history = [
            ChatMessage(role=MessageRole.USER, content="Hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi"),
        ]

        condensed, blocks = build_condensed_history(system_msgs, conv_history)
        assert len(blocks) == 0
        assert len(condensed) == 2

    def test_left_tldr_in_history(self):
        """Test with left TLDR messages."""
        system_msgs = []
        conv_history = [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Summary of past",
                additional_kwargs={"tldr": "left"},
            ),
            ChatMessage(role=MessageRole.USER, content="Current query"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Response"),
        ]

        _, blocks = build_condensed_history(system_msgs, conv_history)
        assert len(blocks) == 1
        assert blocks[0].metadata["tldr_side"] == "left"
        assert blocks[0].metadata["role"] == "assistant"

    def test_right_tldr_in_history(self):
        """Test with right TLDR messages."""
        system_msgs = []
        conv_history = [
            ChatMessage(role=MessageRole.USER, content="Query"),
            ChatMessage(
                role=MessageRole.TOOL,
                content="Tool summary",
                additional_kwargs={"tldr": "right"},
            ),
            ChatMessage(role=MessageRole.ASSISTANT, content="Final response"),
        ]

        _, blocks = build_condensed_history(system_msgs, conv_history)
        assert len(blocks) == 1
        assert blocks[0].metadata["tldr_side"] == "right"
        assert blocks[0].metadata["role"] == "tool"

    def test_mixed_tldr_sides(self):
        """Test with both left and right TLDR messages."""
        system_msgs = []
        conv_history = [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Left summary",
                additional_kwargs={"tldr": "left"},
            ),
            ChatMessage(role=MessageRole.USER, content="Query"),
            ChatMessage(
                role=MessageRole.TOOL,
                content="Right summary",
                additional_kwargs={"tldr": "right"},
            ),
        ]

        _, blocks = build_condensed_history(system_msgs, conv_history)
        assert len(blocks) == 2

        # Find left and right blocks
        left_blocks = [b for b in blocks if b.metadata.get("tldr_side") == "left"]
        right_blocks = [b for b in blocks if b.metadata.get("tldr_side") == "right"]

        assert len(left_blocks) == 1
        assert len(right_blocks) == 1

    def test_system_messages_preserved(self):
        """Test that system messages are preserved."""
        system_msgs = [
            ChatMessage(role=MessageRole.SYSTEM, content="System instruction")
        ]
        conv_history = [
            ChatMessage(role=MessageRole.USER, content="Hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi"),
        ]

        condensed, _ = build_condensed_history(system_msgs, conv_history)
        assert condensed[0].role == MessageRole.SYSTEM
        assert len(condensed) == 3


class TestTLDRSideEdgeCases:
    """Test edge cases for TLDR side functionality."""

    def test_consecutive_left_tldrs(self):
        """Test multiple consecutive left TLDR messages."""
        history = [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Summary 1",
                additional_kwargs={"tldr": "left"},
            ),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Summary 2",
                additional_kwargs={"tldr": "left"},
            ),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Summary 3",
                additional_kwargs={"tldr": "left"},
            ),
            ChatMessage(role=MessageRole.USER, content="Query"),
        ]

        result = trim_to_last_tldr(history)
        # Should keep from the last consecutive TLDR group
        assert len(result) >= 2

    def test_consecutive_right_tldrs(self):
        """Test multiple consecutive right TLDR messages."""
        history = [
            ChatMessage(role=MessageRole.USER, content="Query"),
            ChatMessage(
                role=MessageRole.TOOL,
                content="Tool 1 summary",
                additional_kwargs={"tldr": "right"},
            ),
            ChatMessage(
                role=MessageRole.TOOL,
                content="Tool 2 summary",
                additional_kwargs={"tldr": "right"},
            ),
            ChatMessage(role=MessageRole.ASSISTANT, content="Response"),
        ]

        result = trim_to_last_tldr(history)
        # Should keep from first right TLDR onwards to preserve all messages
        assert len(result) == 4
        assert result[1].content == "Tool 1 summary"
        assert result[2].content == "Tool 2 summary"
        assert result[3].content == "Response"

    def test_tldr_at_start(self):
        """Test TLDR at the very start of conversation."""
        history = [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Initial summary",
                additional_kwargs={"tldr": "left"},
            ),
            ChatMessage(role=MessageRole.USER, content="Query"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Response"),
        ]

        result = trim_to_last_tldr(history)
        assert len(result) == 3

    def test_tldr_at_end(self):
        """Test TLDR at the very end of conversation."""
        history = [
            ChatMessage(role=MessageRole.USER, content="Query"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Response"),
            ChatMessage(
                role=MessageRole.TOOL,
                content="Final summary",
                additional_kwargs={"tldr": "right"},
            ),
        ]

        result = trim_to_last_tldr(history)
        assert len(result) == 2
        assert result[-1].additional_kwargs.get("tldr") == "right"

    def test_only_tldr_messages(self):
        """Test conversation with only TLDR messages."""
        history = [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Summary 1",
                additional_kwargs={"tldr": "left"},
            ),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Summary 2",
                additional_kwargs={"tldr": "left"},
            ),
        ]

        result = trim_to_last_tldr(history)
        # No user message, should return as is
        assert result == history

    def test_alternating_tldr_sides(self):
        """Test alternating left and right TLDR messages."""
        history = [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Left 1",
                additional_kwargs={"tldr": "left"},
            ),
            ChatMessage(role=MessageRole.USER, content="Query 1"),
            ChatMessage(
                role=MessageRole.TOOL,
                content="Right 1",
                additional_kwargs={"tldr": "right"},
            ),
            ChatMessage(role=MessageRole.USER, content="Query 2"),
            ChatMessage(
                role=MessageRole.TOOL,
                content="Right 2",
                additional_kwargs={"tldr": "right"},
            ),
        ]

        result = trim_to_last_tldr(history)
        # Should handle alternating patterns
        assert len(result) >= 2


class TestCondensationInterceptor:
    """Test the condensation interceptor event emission logic."""

    @pytest.mark.asyncio
    async def test_case_1_single_left_side(self):
        """Test Case 1: Emit left block initially, get only left deltas."""

        async def mock_generator():
            # First yield: signal condensation started
            yield CondenseResponse(
                is_condensed=True, chat_history=None, condense_blocks=[]
            )
            # Second yield: provide left-side blocks
            yield CondenseResponse(
                is_condensed=True,
                chat_history=[ChatMessage(role=MessageRole.USER, content="test")],
                condense_blocks=[
                    TextBlock(
                        text="Left summary 1",
                        metadata={"role": "assistant", "tldr_side": "left"},
                    ),
                    TextBlock(
                        text="Left summary 2",
                        metadata={"role": "user", "tldr_side": "left"},
                    ),
                ],
            )

        queue = asyncio.Queue()
        result_container = type(
            "_CondensationResult", (), {"chat_history": None, "condensed": False}
        )()

        await _condensation_producer(queue, result_container, mock_generator())

        # Collect all events
        events = []
        while not queue.empty():
            event = queue.get_nowait()
            if event is not _SENTINEL:  # Skip sentinel
                events.append(event)

        # Verify events
        assert len(events) == 4  # 1 start + 2 deltas + 1 stop

        # Event 0: ContentBlockStart with left side
        assert isinstance(events[0], RawContentBlockStartEvent)
        assert events[0].content_block.tldr_side == "left"

        # Events 1-2: ContentBlockDelta with left side
        assert isinstance(events[1], RawContentBlockDeltaEvent)
        assert isinstance(events[1].delta, TLDRDelta)
        assert events[1].delta.tldr_side == "left"

        assert isinstance(events[2], RawContentBlockDeltaEvent)
        assert isinstance(events[2].delta, TLDRDelta)
        assert events[2].delta.tldr_side == "left"

        # Event 3: ContentBlockStop
        assert isinstance(events[3], RawContentBlockStopEvent)

    @pytest.mark.asyncio
    async def test_case_2_single_right_side(self):
        """Test Case 2: Emit left block initially, get only right deltas."""

        async def mock_generator():
            # First yield: signal condensation started
            yield CondenseResponse(
                is_condensed=True, chat_history=None, condense_blocks=[]
            )
            # Second yield: provide right-side blocks only
            yield CondenseResponse(
                is_condensed=True,
                chat_history=[ChatMessage(role=MessageRole.USER, content="test")],
                condense_blocks=[
                    TextBlock(
                        text="Right summary 1",
                        metadata={"role": "tool", "tldr_side": "right"},
                    ),
                    TextBlock(
                        text="Right summary 2",
                        metadata={"role": "tool", "tldr_side": "right"},
                    ),
                ],
            )

        queue = asyncio.Queue()
        result_container = type(
            "_CondensationResult", (), {"chat_history": None, "condensed": False}
        )()

        await _condensation_producer(queue, result_container, mock_generator())

        # Collect all events
        events = []
        while not queue.empty():
            event = queue.get_nowait()
            if event is not _SENTINEL:  # Skip sentinel
                events.append(event)

        # Verify events - should only have 1 block (reused)
        assert len(events) == 4  # 1 start + 2 deltas + 1 stop

        # Event 0: ContentBlockStart with left side (initial)
        assert isinstance(events[0], RawContentBlockStartEvent)
        assert events[0].content_block.tldr_side == "left"
        initial_block_id = events[0].block_id

        # Events 1-2: ContentBlockDelta with RIGHT side (overrides)
        assert isinstance(events[1], RawContentBlockDeltaEvent)
        assert events[1].block_id == initial_block_id  # Same block reused
        assert isinstance(events[1].delta, TLDRDelta)
        assert events[1].delta.tldr_side == "right"  # Side changed

        assert isinstance(events[2], RawContentBlockDeltaEvent)
        assert events[2].block_id == initial_block_id  # Same block reused
        assert isinstance(events[2].delta, TLDRDelta)
        assert events[2].delta.tldr_side == "right"

        # Event 3: ContentBlockStop
        assert isinstance(events[3], RawContentBlockStopEvent)

    @pytest.mark.asyncio
    async def test_case_3_both_sides(self):
        """Test Case 3: Emit left block initially, get both left and right deltas."""

        async def mock_generator():
            # First yield: signal condensation started
            yield CondenseResponse(
                is_condensed=True, chat_history=None, condense_blocks=[]
            )
            # Second yield: provide both left and right blocks
            yield CondenseResponse(
                is_condensed=True,
                chat_history=[ChatMessage(role=MessageRole.USER, content="test")],
                condense_blocks=[
                    TextBlock(
                        text="Left summary",
                        metadata={"role": "assistant", "tldr_side": "left"},
                    ),
                    TextBlock(
                        text="Right summary 1",
                        metadata={"role": "tool", "tldr_side": "right"},
                    ),
                    TextBlock(
                        text="Right summary 2",
                        metadata={"role": "tool", "tldr_side": "right"},
                    ),
                ],
            )

        queue = asyncio.Queue()
        result_container = type(
            "_CondensationResult", (), {"chat_history": None, "condensed": False}
        )()

        await _condensation_producer(queue, result_container, mock_generator())

        # Collect all events
        events = []
        while not queue.empty():
            event = queue.get_nowait()
            if event is not _SENTINEL:  # Skip sentinel
                events.append(event)

        # Verify events - should have 2 blocks
        assert len(events) == 7  # 2 starts + 3 deltas + 2 stops

        # Event 0: ContentBlockStart with left side (initial)
        assert isinstance(events[0], RawContentBlockStartEvent)
        assert events[0].content_block.tldr_side == "left"
        left_block_id = events[0].block_id

        # Event 1: ContentBlockDelta for left block
        assert isinstance(events[1], RawContentBlockDeltaEvent)
        assert events[1].block_id == left_block_id
        assert isinstance(events[1].delta, TLDRDelta)
        assert events[1].delta.tldr_side == "left"

        # Event 2: ContentBlockStart with right side (new block)
        assert isinstance(events[2], RawContentBlockStartEvent)
        assert events[2].content_block.tldr_side == "right"
        right_block_id = events[2].block_id
        assert right_block_id != left_block_id  # Different block

        # Events 3-4: ContentBlockDelta for right block
        assert isinstance(events[3], RawContentBlockDeltaEvent)
        assert events[3].block_id == right_block_id
        assert isinstance(events[3].delta, TLDRDelta)
        assert events[3].delta.tldr_side == "right"

        assert isinstance(events[4], RawContentBlockDeltaEvent)
        assert events[4].block_id == right_block_id
        assert isinstance(events[4].delta, TLDRDelta)
        assert events[4].delta.tldr_side == "right"

        # Events 5-6: ContentBlockStop for both blocks
        assert isinstance(events[5], RawContentBlockStopEvent)
        assert isinstance(events[6], RawContentBlockStopEvent)

    @pytest.mark.asyncio
    async def test_no_condensation(self):
        """Test when condensation doesn't happen (is_condensed=False)."""

        async def mock_generator():
            yield CondenseResponse(
                is_condensed=False, chat_history=None, condense_blocks=None
            )

        queue = asyncio.Queue()
        result_container = type(
            "_CondensationResult", (), {"chat_history": None, "condensed": False}
        )()

        await _condensation_producer(queue, result_container, mock_generator())

        # Collect all events
        events = []
        while not queue.empty():
            event = queue.get_nowait()
            if event is not _SENTINEL:  # Skip sentinel
                events.append(event)

        # Should only have sentinel, no blocks emitted
        assert len(events) == 0
        assert result_container.condensed is False

    @pytest.mark.asyncio
    async def test_empty_condense_blocks(self):
        """Test initial feedback emission with empty condense_blocks."""

        async def mock_generator():
            # First yield: signal condensation started
            yield CondenseResponse(
                is_condensed=True, chat_history=None, condense_blocks=[]
            )
            # Second yield: no actual blocks (edge case)
            yield CondenseResponse(
                is_condensed=True,
                chat_history=[ChatMessage(role=MessageRole.USER, content="test")],
                condense_blocks=[],
            )

        queue = asyncio.Queue()
        result_container = type(
            "_CondensationResult", (), {"chat_history": None, "condensed": False}
        )()

        await _condensation_producer(queue, result_container, mock_generator())

        # Collect all events
        events = []
        while not queue.empty():
            event = queue.get_nowait()
            if event is not _SENTINEL:  # Skip sentinel
                events.append(event)

        # Should emit initial block and stop
        assert len(events) == 2  # 1 start + 1 stop
        assert isinstance(events[0], RawContentBlockStartEvent)
        assert events[0].content_block.tldr_side == "left"
        assert isinstance(events[1], RawContentBlockStopEvent)

    @pytest.mark.asyncio
    async def test_multiple_yields_same_side(self):
        """Test multiple yields adding more deltas to the same side."""

        async def mock_generator():
            # First yield: signal condensation started
            yield CondenseResponse(
                is_condensed=True, chat_history=None, condense_blocks=[]
            )
            # Second yield: provide left blocks
            yield CondenseResponse(
                is_condensed=True,
                chat_history=[ChatMessage(role=MessageRole.USER, content="test")],
                condense_blocks=[
                    TextBlock(
                        text="Left summary 1",
                        metadata={"role": "assistant", "tldr_side": "left"},
                    ),
                ],
            )

        queue = asyncio.Queue()
        result_container = type(
            "_CondensationResult", (), {"chat_history": None, "condensed": False}
        )()

        await _condensation_producer(queue, result_container, mock_generator())

        # Collect all events
        events = []
        while not queue.empty():
            event = queue.get_nowait()
            if event is not _SENTINEL:  # Skip sentinel
                events.append(event)

        # Verify proper event sequence
        assert len(events) == 3  # 1 start + 1 delta + 1 stop
        assert isinstance(events[0], RawContentBlockStartEvent)
        assert isinstance(events[1], RawContentBlockDeltaEvent)
        assert isinstance(events[2], RawContentBlockStopEvent)
        assert result_container.condensed is True
