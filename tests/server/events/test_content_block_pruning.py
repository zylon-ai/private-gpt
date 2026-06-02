from typing import Literal

import pytest

from private_gpt.events.models import (
    AudioBlock,
    BinaryBlock,
    ImageBlock,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    ResourceBlock,
    ResourceLinkBlock,
    SourceBlock,
    SourceDelta,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    TLDRBlock,
    TLDRDelta,
    ToolResultBlock,
    ToolUseBlock,
)


class TestStandardContentBlocks:
    @pytest.mark.parametrize("response_mode", ["anthropic", "zylon"])
    def test_text_block_pruning(self, response_mode: Literal["anthropic", "zylon"]):
        block = TextBlock(text="Hello, world!")
        result = block.prune_content_block_by_response_mode(response_mode)

        assert result is not None
        assert isinstance(result, TextBlock)
        assert result.text == "Hello, world!"

    @pytest.mark.parametrize("response_mode", ["anthropic", "zylon"])
    def test_image_block_pruning(self, response_mode: Literal["anthropic", "zylon"]):
        block = ImageBlock.from_base64(data="base64data", mime_type="image/png")
        result = block.prune_content_block_by_response_mode(response_mode)

        assert result is not None
        assert isinstance(result, ImageBlock)
        assert result.source.data == "base64data"

    @pytest.mark.parametrize("response_mode", ["anthropic", "zylon"])
    def test_audio_block_pruning(self, response_mode: Literal["anthropic", "zylon"]):
        block = AudioBlock.from_base64(data="base64audiodata", mime_type="audio/mpeg")
        result = block.prune_content_block_by_response_mode(response_mode)

        assert result is not None
        assert isinstance(result, AudioBlock)
        assert result.source.media_type == "audio/mpeg"

    @pytest.mark.parametrize("response_mode", ["anthropic", "zylon"])
    def test_resource_link_block_pruning(
        self, response_mode: Literal["anthropic", "zylon"]
    ):
        block = ResourceLinkBlock(
            uri="https://example.com/resource", name="Example Resource"
        )
        result = block.prune_content_block_by_response_mode(response_mode)

        assert result is not None
        assert isinstance(result, ResourceLinkBlock)
        assert result.uri == "https://example.com/resource"

    @pytest.mark.parametrize("response_mode", ["anthropic", "zylon"])
    def test_resource_block_pruning(self, response_mode: Literal["anthropic", "zylon"]):
        block = ResourceBlock(
            resource=ResourceBlock.Resource(
                uri="https://example.com/doc", name="Example Document"
            )
        )
        result = block.prune_content_block_by_response_mode(response_mode)

        assert result is not None
        assert isinstance(result, ResourceBlock)
        assert result.resource.name == "Example Document"

    @pytest.mark.parametrize("response_mode", ["anthropic", "zylon"])
    def test_tool_use_block_pruning(self, response_mode: Literal["anthropic", "zylon"]):
        block = ToolUseBlock(id="tool_123", name="search_tool", input={"query": "test"})
        result = block.prune_content_block_by_response_mode(response_mode)

        assert result is not None
        assert isinstance(result, ToolUseBlock)
        assert result.name == "search_tool"

    @pytest.mark.parametrize("response_mode", ["anthropic", "zylon"])
    def test_thinking_block_pruning(self, response_mode: Literal["anthropic", "zylon"]):
        block = ThinkingBlock(
            thinking="Let me think about this...", signature="sig_test_123"
        )
        result = block.prune_content_block_by_response_mode(response_mode)

        assert result is not None
        assert isinstance(result, ThinkingBlock)
        assert result.thinking == "Let me think about this..."


class TestExtendedContentBlocks:
    def test_binary_block_pruning_anthropic_mode(self):
        block = BinaryBlock(
            filename="document.pdf",
            source={
                "type": "base64",
                "data": "base64binarydata",
                "media_type": "application/pdf",
            },
        )
        result = block.prune_content_block_by_response_mode("anthropic")

        assert result is None

    def test_binary_block_pruning_zylon_mode(self):
        block = BinaryBlock(
            filename="document.pdf",
            source={
                "type": "base64",
                "data": "base64binarydata",
                "media_type": "application/pdf",
            },
        )
        result = block.prune_content_block_by_response_mode("zylon")

        assert result is not None
        assert isinstance(result, BinaryBlock)
        assert result.filename == "document.pdf"

    def test_source_block_pruning_anthropic_mode(self):
        block = SourceBlock(sources=[])
        result = block.prune_content_block_by_response_mode("anthropic")

        assert result is None

    def test_source_block_pruning_zylon_mode(self):
        block = SourceBlock(sources=[])
        result = block.prune_content_block_by_response_mode("zylon")

        assert result is not None
        assert isinstance(result, SourceBlock)

    def test_tldr_block_pruning_anthropic_mode(self):
        block = TLDRBlock(content=[TextBlock(text="Summary")])
        result = block.prune_content_block_by_response_mode("anthropic")

        assert result is None

    def test_tldr_block_pruning_zylon_mode(self):
        block = TLDRBlock(content=[TextBlock(text="Summary")])
        result = block.prune_content_block_by_response_mode("zylon")

        assert result is not None
        assert isinstance(result, TLDRBlock)
        assert len(result.content) == 1


class TestToolResultBlockPruning:
    def test_tool_result_with_string_content(self):
        block = ToolResultBlock(tool_use_id="tool_123", content="Simple string result")

        result_anthropic = block.prune_content_block_by_response_mode("anthropic")
        result_zylon = block.prune_content_block_by_response_mode("zylon")

        assert result_anthropic is not None
        assert result_zylon is not None
        assert isinstance(result_anthropic.content, str)
        assert isinstance(result_zylon.content, str)

    def test_tool_result_with_mixed_content_anthropic_mode(self):
        block = ToolResultBlock(
            tool_use_id="tool_123",
            content=[
                TextBlock(text="Standard text"),
                SourceBlock(sources=[]),
                ImageBlock.from_base64(data="img", mime_type="image/png"),
            ],
        )

        result = block.prune_content_block_by_response_mode("anthropic")

        assert result is not None
        assert isinstance(result.content, list)
        assert len(result.content) == 2
        assert isinstance(result.content[0], TextBlock)
        assert isinstance(result.content[1], ImageBlock)

    def test_tool_result_with_mixed_content_zylon_mode(self):
        block = ToolResultBlock(
            tool_use_id="tool_123",
            content=[
                TextBlock(text="Standard text"),
                SourceBlock(sources=[]),
                ImageBlock.from_base64(data="img", mime_type="image/png"),
            ],
        )

        result = block.prune_content_block_by_response_mode("zylon")

        assert result is not None
        assert isinstance(result.content, list)
        assert len(result.content) == 3
        assert isinstance(result.content[0], TextBlock)
        assert isinstance(result.content[1], SourceBlock)
        assert isinstance(result.content[2], ImageBlock)

    def test_tool_result_with_only_extended_content_anthropic_mode(self):
        block = ToolResultBlock(
            tool_use_id="tool_123",
            content=[
                SourceBlock(sources=[]),
                TLDRBlock(content=[TextBlock(text="Summary")]),
            ],
        )

        result = block.prune_content_block_by_response_mode("anthropic")

        assert result is None


class TestStreamingEventPruning:
    def test_content_block_start_with_standard_block(self):
        start = RawContentBlockStartEvent(
            block_id="block_123", content_block=TextBlock(text="Starting...")
        )

        result_anthropic = start.prune_content_block_by_response_mode("anthropic")
        result_zylon = start.prune_content_block_by_response_mode("zylon")

        assert result_anthropic is not None
        assert result_zylon is not None

    def test_content_block_start_with_extended_block_anthropic_mode(self):
        start = RawContentBlockStartEvent(
            block_id="block_123", content_block=SourceBlock(sources=[])
        )

        result = start.prune_content_block_by_response_mode("anthropic")

        assert result is None

    def test_content_block_start_with_extended_block_zylon_mode(self):
        start = RawContentBlockStartEvent(
            block_id="block_123", content_block=SourceBlock(sources=[])
        )

        result = start.prune_content_block_by_response_mode("zylon")

        assert result is not None
        assert isinstance(result.content_block, SourceBlock)

    def test_content_block_delta_with_standard_delta(self):
        delta = RawContentBlockDeltaEvent(
            block_id="block_123", delta=TextDelta(text="more text...")
        )

        result_anthropic = delta.prune_content_block_by_response_mode("anthropic")
        result_zylon = delta.prune_content_block_by_response_mode("zylon")

        assert result_anthropic is not None
        assert result_zylon is not None

    def test_content_block_delta_with_extended_delta_anthropic_mode(self):
        delta = RawContentBlockDeltaEvent(
            block_id="block_123", delta=SourceDelta(sources=[])
        )

        result = delta.prune_content_block_by_response_mode("anthropic")

        assert result is None

    def test_content_block_delta_with_extended_delta_zylon_mode(self):
        delta = RawContentBlockDeltaEvent(
            block_id="block_123", delta=SourceDelta(sources=[])
        )

        result = delta.prune_content_block_by_response_mode("zylon")

        assert result is not None
        assert isinstance(result.delta, SourceDelta)


class TestDeltaBlockPruning:
    @pytest.mark.parametrize("response_mode", ["anthropic", "zylon"])
    def test_text_delta_pruning(self, response_mode: Literal["anthropic", "zylon"]):
        delta = TextDelta(text="delta text")
        result = delta.prune_content_block_by_response_mode(response_mode)

        assert result is not None
        assert isinstance(result, TextDelta)

    def test_source_delta_pruning_anthropic_mode(self):
        delta = SourceDelta(sources=[])
        result = delta.prune_content_block_by_response_mode("anthropic")

        assert result is None

    def test_source_delta_pruning_zylon_mode(self):
        delta = SourceDelta(sources=[])
        result = delta.prune_content_block_by_response_mode("zylon")

        assert result is not None
        assert isinstance(result, SourceDelta)

    def test_tldr_delta_pruning_anthropic_mode(self):
        delta = TLDRDelta(tldr=TextBlock(text="Summary"))
        result = delta.prune_content_block_by_response_mode("anthropic")

        assert result is None

    def test_tldr_delta_pruning_zylon_mode(self):
        delta = TLDRDelta(tldr=TextBlock(text="Summary"))
        result = delta.prune_content_block_by_response_mode("zylon")

        assert result is not None
        assert isinstance(result, TLDRDelta)
