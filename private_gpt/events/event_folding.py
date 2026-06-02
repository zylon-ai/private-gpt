from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from private_gpt.chat.extensions.citation import ZylonCitation
from private_gpt.components.chunk.models import SourceType
from private_gpt.events.models import (
    ContentBlockType,
    Event,
    FatalError,
    InputJSONDelta,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    SourceBlock,
    SourceDelta,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    TLDRBlock,
    TLDRDelta,
    ToolUseBlock,
    Usage,
)

if TYPE_CHECKING:
    from private_gpt.events.models import BasicContentBlockType


class ChatResponse(BaseModel):
    content: list[ContentBlockType] = Field(
        default_factory=list, description="Content blocks"
    )
    exception: BaseException | None = Field(
        default=None, description="Exception if any"
    )
    stop_reason: str | None = Field(default=None, description="Finish reason")
    usage: Usage | None = Field(default=None, description="Usage information")

    class Config:
        arbitrary_types_allowed = True

    @property
    def response(self) -> str | None:
        """Get the response from the content blocks."""
        if not self.content:
            return None
        text_block = [
            content for content in self.content if isinstance(content, TextBlock)
        ]
        return text_block[0].text if text_block else None

    @property
    def sources(self) -> list[SourceType] | None:
        """Get the sources from the content blocks."""
        if not self.content:
            return None
        return [
            source
            for content in self.content
            if isinstance(content, SourceBlock)
            for source in content.sources
        ]

    @property
    def citations(self) -> list[ZylonCitation] | None:
        """Get the citations from the content blocks."""
        if not self.content:
            return None
        return [
            citation
            for content in self.content
            if isinstance(content, TextBlock)
            for citation in content.citations or []
        ]


async def fold_events(
    streaming_handler: AsyncGenerator[Event | Exception | None, None],
) -> AsyncGenerator[ChatResponse, None]:
    response = ChatResponse(content=[])
    content_blocks: dict[str, ContentBlockType] = {}

    async for event in streaming_handler:
        if not event:
            continue

        elif isinstance(event, Exception):
            response.content = []
            response.exception = event
            yield response
            break

        elif isinstance(event, FatalError):
            response.content = []
            response.exception = event.exception
            yield response
            break

        elif isinstance(event, RawContentBlockStartEvent) and event.content_block:
            if event.block_id not in content_blocks:
                content_blocks[event.block_id] = event.content_block
                response.content.append(event.content_block)

        elif isinstance(event, RawContentBlockDeltaEvent) and event.delta:
            if isinstance(event.delta, TextDelta) and event.block_id in content_blocks:
                text_block = content_blocks[event.block_id]
                assert isinstance(text_block, TextBlock)
                if text_block.text is None:
                    text_block.text = ""
                text_block.text += event.delta.text or ""
                if text_block.citations is None and event.delta.citations is not None:
                    text_block.citations = []
                if (
                    text_block.citations is not None
                    and event.delta.citations is not None
                ):
                    text_block.citations.extend(event.delta.citations)
            elif (
                isinstance(event.delta, SourceDelta)
                and event.block_id in content_blocks
            ):
                source_block = content_blocks[event.block_id]
                assert isinstance(source_block, SourceBlock)
                if source_block.sources is None:
                    source_block.sources = []
                source_block.sources += event.delta.sources
            elif (
                isinstance(event.delta, InputJSONDelta)
                and event.block_id in content_blocks
            ):
                block = content_blocks[event.block_id]
                if isinstance(block, ToolUseBlock):
                    if block.input is None:
                        block.input = {}
                    block.input = event.delta.partial_json_obj
                else:
                    raise TypeError(
                        f"Unexpected block type {type(block)} for InputJSONDelta"
                    )
            elif (
                isinstance(event.delta, ThinkingDelta)
                and event.block_id in content_blocks
            ):
                thinking_block = content_blocks[event.block_id]
                assert isinstance(thinking_block, ThinkingBlock)
                if thinking_block.thinking is None:
                    thinking_block.thinking = ""
                thinking_block.thinking += event.delta.thinking or ""
                if (
                    thinking_block.citations is None
                    and event.delta.citations is not None
                ):
                    thinking_block.citations = []
                if (
                    thinking_block.citations is not None
                    and event.delta.citations is not None
                ):
                    thinking_block.citations.extend(event.delta.citations)

            elif (
                isinstance(event.delta, TLDRDelta) and event.block_id in content_blocks
            ):
                tldr_block = content_blocks[event.block_id]
                assert isinstance(tldr_block, TLDRBlock)
                if tldr_block.content is None:
                    tldr_block.content = []
                delta: BasicContentBlockType = event.delta.tldr
                tldr_block.content.append(delta)

        elif isinstance(event, RawContentBlockStopEvent):
            if event.block_id in content_blocks:
                content_block = content_blocks[event.block_id]
                if content_block.start_timestamp and event.stop_timestamp:
                    content_block.stop_timestamp = event.stop_timestamp

        elif isinstance(event, RawMessageDeltaEvent):
            if event.delta:
                response.stop_reason = event.delta.stop_reason
            if event.usage:
                response.usage = event.usage

        yield response


async def fold(
    streaming_handler: AsyncGenerator[Event | Exception | None, None],
) -> ChatResponse:
    response = ChatResponse(content=[])
    async for event in fold_events(streaming_handler):
        response = event
    return response
