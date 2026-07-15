from collections.abc import Iterator
from contextlib import contextmanager
from uuid import uuid4

from private_gpt.components.chunk.models import Chunk
from private_gpt.events.models import (
    Event,
    FatalError,
    MessageOutputDelta,
    PingEvent,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    SourceDelta,
    Usage,
)
from private_gpt.events.sse.sse_manager import SSEStreamManager


class SSEProducer:
    manager: SSEStreamManager

    def __init__(
        self,
        model_name: str,
        manager: SSEStreamManager | None = None,
    ) -> None:
        self._model_name = model_name
        self._input_token_count = 0
        self._output_token_count = 0
        self._content_block_block_id = self._generate_block_id()

        self.manager = manager or SSEStreamManager()

    def send_ping(self) -> None:
        self._send_event(PingEvent())

    def _generate_message_id(self) -> str:
        return f"msg_{uuid4().hex}"

    def _generate_block_id(self) -> str:
        return f"block_{uuid4().hex}"

    @contextmanager
    def message_stream(self) -> Iterator[RawMessageStartEvent]:
        message_start = RawMessageStartEvent.from_defaults()
        message_start.message.id = self._generate_message_id()
        message_start.message.model = self._model_name
        message_start.message.usage.input_tokens = (
            self._input_token_count if self._input_token_count > 0 else None
        )
        self._send_event(message_start)

        try:
            yield message_start
            self._send_event(RawMessageStopEvent())
        except Exception as e:
            self._send_event(FatalError.from_exception(e))
        finally:
            self._send_event(None)

    @contextmanager
    def content_block(self) -> Iterator[RawContentBlockStartEvent]:
        block_start = RawContentBlockStartEvent.from_text(
            block_id=self._content_block_block_id
        )
        self._send_event(block_start)

        try:
            yield block_start
        finally:
            self._send_event(RawContentBlockStopEvent(block_id=block_start.block_id))
            self._content_block_block_id = self._generate_block_id()

    def process_content_blocks(
        self,
        generator: Iterator[Event | Exception | None],
    ) -> None:
        for block in generator:
            if isinstance(block, Exception):
                raise block

            if block is None:
                continue

            # Send the content block delta
            self._send_event(block)

    def process_sources(
        self, start: RawContentBlockStartEvent, sources: list[Chunk]
    ) -> None:
        if not sources:
            return

        source_delta = SourceDelta(sources=sources)
        self._send_event(
            RawContentBlockDeltaEvent(block_id=start.block_id, delta=source_delta)
        )

    def set_end_message(self, stop_reason: str = "end_turn") -> None:
        self._send_event(
            RawMessageDeltaEvent(
                delta=MessageOutputDelta(stop_reason=stop_reason),
                usage=Usage(
                    input_tokens=self._input_token_count
                    if self._input_token_count > 0
                    else None,
                    output_tokens=self._output_token_count
                    if self._output_token_count > 0
                    else None,
                ),
            )
        )

    def _send_event(self, event: Event | None) -> None:
        self.manager.send_event(event)

    def __enter__(self) -> "SSEProducer":
        return self

    def __exit__(self, exc_type: Exception, exc_value: str, traceback: str) -> None:
        self.close()

    def close(self) -> None:
        self.manager.close()
        self._input_token_count = 0
        self._output_token_count = 0
        self._content_block_block_id = self._generate_block_id()

    async def aclose(self) -> None:
        await self.manager.aclose()
        self._input_token_count = 0
        self._output_token_count = 0
        self._content_block_block_id = self._generate_block_id()
