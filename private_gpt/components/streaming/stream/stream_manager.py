import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from injector import inject, singleton
from pydantic import BaseModel

from private_gpt.components.streaming.providers.models import StreamMetadata
from private_gpt.components.streaming.stream.event_handler import EventHandler
from private_gpt.components.streaming.stream.stream_processor import StreamProcessor
from private_gpt.components.streaming.stream.stream_reader import (
    AdaptiveStreamReader,
    StreamReader,
)
from private_gpt.components.streaming.stream_component import StreamComponent
from private_gpt.settings.settings import Settings


@singleton
class StreamManager:
    """Main interface for stream operations."""

    @inject
    def __init__(
        self,
        settings: Settings,
        stream_component: StreamComponent,
        stream_processor: StreamProcessor,
        stream_reader: StreamReader,
    ):
        self.stream_service = stream_component.stream
        self.processor = stream_processor
        self.reader: StreamReader | AdaptiveStreamReader = stream_reader
        if settings.chat.multiplexing_threshold:
            self.reader = AdaptiveStreamReader(settings, stream_reader)

    async def create_and_start_stream(
        self,
        stream_type: str,
        event_generator: AsyncGenerator[Any, None],
        event_handler: EventHandler,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a stream and start processing it."""
        correlation_id = await self.stream_service.create_stream(
            stream_type=stream_type,
            correlation_id=correlation_id,
            metadata=metadata,
        )

        await self.processor.start_stream_processing(
            event_handler=event_handler,
            correlation_id=correlation_id,
            stream_type=stream_type,
            event_generator=event_generator,
            metadata=metadata,
        )

        return correlation_id

    async def create_stream(
        self,
        stream_type: str,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return await self.stream_service.create_stream(
            stream_type=stream_type,
            correlation_id=correlation_id,
            metadata=metadata,
        )

    async def cancel_stream(self, correlation_id: str) -> bool:
        """Cancel a stream."""
        return await self.processor.cancel_stream_processing(correlation_id)

    async def get_stream_metadata(self, correlation_id: str) -> StreamMetadata | None:
        """Get stream metadata."""
        return await self.stream_service.get_stream_metadata(correlation_id)

    async def stream_exists(self, correlation_id: str) -> bool:
        """Check if stream exists."""
        return await self.stream_service.stream_exists(correlation_id)

    async def read_events(
        self,
        event_handler: EventHandler,
        correlation_id: str,
        last_id: str = "0",
        count: int = 100,
    ) -> list[BaseModel]:
        """Read events as list of deserialized objects."""
        events, last_id = await self.reader.read_events(
            event_handler=event_handler,
            correlation_id=correlation_id,
            last_id=last_id,
            count=count,
        )
        return events

    async def stream_events(
        self,
        event_handler: EventHandler,
        correlation_id: str,
        last_id: str = "0",
        stop_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[Any, None]:
        """Stream events as they arrive."""
        async for event in await self.reader.stream_events(
            event_handler=event_handler,
            correlation_id=correlation_id,
            last_id=last_id,
            stop_event=stop_event,
        ):
            yield event

    async def clean_up_stream(self, correlation_id: str) -> None:
        """Clean up a specific stream."""
        await self.cancel_stream(correlation_id)
        await self.stream_service.clean_up_stream(correlation_id)

    async def cleanup(self) -> None:
        """Clean up all resources."""
        await self.stream_service.close()
