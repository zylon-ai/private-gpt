from collections.abc import AsyncGenerator

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ChatRequest
from private_gpt.components.streaming.providers.models import StreamMetadata
from private_gpt.components.streaming.stream.stream_manager import StreamManager
from private_gpt.events.event_serializer import StreamingEventHandler
from private_gpt.events.models import Event
from private_gpt.server.chat.chat_facade import ChatFacadeService


@singleton
class ChatAsyncService:
    @inject
    def __init__(self, chat_facade: ChatFacadeService, stream_manager: StreamManager):
        self._chat_facade = chat_facade
        self.stream_manager = stream_manager

    async def initiate_chat_stream(
        self, request: ChatRequest, message_id: str | None = None
    ) -> str:
        """Initiate a chat completion stream.

        Returns:
            Tuple of (message_id, message)

        Raises:
            HTTPException: If message_id already exists
        """
        if message_id and await self.stream_manager.stream_exists(message_id):
            raise ValueError("Stream with this message_id already exists")

        event_generator = await self._chat_facade.create_chat_event_generator(
            request=request
        )
        message_id = await self.stream_manager.create_and_start_stream(
            event_handler=StreamingEventHandler(),
            stream_type="chat_completion",
            event_generator=event_generator,
            correlation_id=message_id,
            # For debugging purposes, we can include metadata about the request
            metadata={
                "message_count": len(request.messages),
                "thinking_enabled": request.thinking.enabled,
                "citations_enabled": request.citation.enabled,
            },
        )

        return message_id

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
