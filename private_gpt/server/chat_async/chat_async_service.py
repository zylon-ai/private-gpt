from collections.abc import AsyncGenerator

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ChatRequest
from private_gpt.components.streaming.providers.models import StreamMetadata
from private_gpt.components.streaming.stream.stream_manager import StreamManager
from private_gpt.events.event_serializer import StreamingEventHandler
from private_gpt.events.models import Event
from private_gpt.server.chat_async.chat_worker_dispatch import (
    create_stream_and_enqueue_chat,
)


@singleton
class ChatAsyncService:
    @inject
    def __init__(self, stream_manager: StreamManager):
        self.stream_manager = stream_manager

    async def initiate_chat_stream(
        self, request: ChatRequest, message_id: str | None = None
    ) -> str:
        """Initiate a chat completion stream."""
        if message_id and await self.stream_manager.stream_exists(message_id):
            raise ValueError("Stream with this message_id already exists")

        return await create_stream_and_enqueue_chat(
            stream_manager=self.stream_manager,
            request=request,
            message_id=message_id,
        )

    async def get_stream_events(
        self, message_id: str
    ) -> AsyncGenerator[Event, None] | None:
        """Get Events stream for a chat completion."""
        if not await self.stream_manager.stream_exists(message_id):
            return None

        return self.stream_manager.stream_events(
            event_handler=StreamingEventHandler(), correlation_id=message_id
        )

    async def cancel_stream(self, message_id: str) -> bool | None:
        """Cancel an active chat stream."""
        return await self.stream_manager.cancel_stream(message_id)

    async def clean_up_stream(self, message_id: str) -> None:
        """Cleanup resources associated with a stream."""
        await self.stream_manager.clean_up_stream(message_id)

    async def get_stream_metadata(self, message_id: str) -> StreamMetadata | None:
        """Get stream status and metadata."""
        return await self.stream_manager.get_stream_metadata(message_id)
