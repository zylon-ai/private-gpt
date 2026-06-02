import datetime
from typing import Annotated, Literal, Self
from uuid import uuid4

from pydantic import Field

from private_gpt.events.models._base import BaseContentBlock, StandardContentProtocol
from private_gpt.events.models._deltas import ContentBlockDeltaType
from private_gpt.events.models._errors import FatalError
from private_gpt.events.models._message import Message, MessageOutputDelta, Usage
from private_gpt.events.models._tool_result_blocks import ContentBlockType


class RawContentBlockStartEvent(BaseContentBlock, StandardContentProtocol):
    """Signals the start of a new content block during streaming."""

    type: Literal["content_block_start"] = Field(default="content_block_start")
    index: int | None = Field(default=None)
    block_id: str = Field(description="Zylon-internal unique identifier for this block")
    content_block: ContentBlockType = Field(
        description="The initial (possibly empty) content block"
    )

    @classmethod
    def from_text(
        cls, block_id: str | None = None, text: str = ""
    ) -> "RawContentBlockStartEvent":
        from private_gpt.events.models._content_blocks import TextBlock

        return cls(
            block_id=block_id or f"block_{uuid4().hex}",
            content_block=TextBlock(text=text),
        )

    def prune_content_block_by_response_mode(
        self, response_mode: Literal["anthropic", "zylon"]
    ) -> Self | None:
        if self.content_block.prune_content_block_by_response_mode(response_mode):
            return self
        return None


class RawContentBlockDeltaEvent(BaseContentBlock, StandardContentProtocol):
    """Carries an incremental delta for an in-progress content block."""

    type: Literal["content_block_delta"] = Field(default="content_block_delta")
    index: int | None = Field(default=None)
    block_id: str = Field(
        description="Matches the block_id of the originating start event"
    )
    delta: ContentBlockDeltaType = Field(description="The incremental update")

    @classmethod
    def from_content_block_start(
        cls,
        start: RawContentBlockStartEvent,
        delta: ContentBlockDeltaType,
    ) -> "RawContentBlockDeltaEvent":
        return cls(index=start.index, block_id=start.block_id, delta=delta)

    def prune_content_block_by_response_mode(
        self, response_mode: Literal["anthropic", "zylon"]
    ) -> Self | None:
        if self.delta.prune_content_block_by_response_mode(response_mode):
            return self
        return None


class RawContentBlockStopEvent(BaseContentBlock, StandardContentProtocol):
    """Signals that a content block has finished streaming."""

    type: Literal["content_block_stop"] = Field(default="content_block_stop")
    stop_timestamp: datetime.datetime | None = Field(
        default=None, serialization_alias="stop_timestamp"
    )
    index: int | None = Field(default=None)
    block_id: str = Field(description="Matches the originating start event's block_id")

    @classmethod
    def from_start(cls, start: RawContentBlockStartEvent) -> "RawContentBlockStopEvent":
        return cls(
            index=start.index,
            block_id=start.block_id,
            stop_timestamp=datetime.datetime.now().astimezone(),
        )


class RawMessageStartEvent(BaseContentBlock, StandardContentProtocol):
    """Opens a streaming message response."""

    type: Literal["message_start"] = Field(default="message_start")
    message: Message = Field(default_factory=Message)

    @classmethod
    def from_defaults(cls) -> "RawMessageStartEvent":
        return cls(message=Message(usage=Usage(input_tokens=0, output_tokens=0)))


class RawMessageDeltaEvent(BaseContentBlock, StandardContentProtocol):
    """Carries partial message-level metadata updates (stop_reason, usage, …)."""

    type: Literal["message_delta"] = Field(default="message_delta")
    delta: MessageOutputDelta | None = Field(default=None)
    usage: Usage | None = Field(default=None)

    @classmethod
    def from_defaults(cls) -> "RawMessageDeltaEvent":
        return cls(delta=MessageOutputDelta(), usage=Usage())

    def update(self, delta: RawContentBlockDeltaEvent) -> None:
        pass


class RawMessageStopEvent(BaseContentBlock, StandardContentProtocol):
    """Signals the end of the streaming message."""

    type: Literal["message_stop"] = Field(default="message_stop")

    @classmethod
    def from_defaults(cls) -> "RawMessageStopEvent":
        return cls()


class PingEvent(BaseContentBlock, StandardContentProtocol):
    """SSE keepalive ping."""

    type: Literal["ping"] = Field(default="ping")


Event = Annotated[
    RawContentBlockStartEvent
    | RawContentBlockDeltaEvent
    | RawContentBlockStopEvent
    | RawMessageStartEvent
    | RawMessageDeltaEvent
    | RawMessageStopEvent
    | PingEvent
    | FatalError,
    Field(discriminator="type"),
]
