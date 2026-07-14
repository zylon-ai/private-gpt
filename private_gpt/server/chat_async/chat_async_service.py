from collections.abc import AsyncGenerator
from uuid import uuid4

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ChatRequest
from private_gpt.components.streaming.providers.models import StreamMetadata
from private_gpt.components.streaming.stream.stream_manager import StreamManager
from private_gpt.events.event_serializer import StreamingEventHandler
from private_gpt.events.interceptors.ping_event_interceptor import (
    PingEventInterceptor,
)
from private_gpt.events.models import Event
from private_gpt.server.chat.chat_facade import ChatFacadeService


@singleton
class ChatAsyncService:
    @inject
    def __init__(
        self,
        stream_manager: StreamManager,
        chat_facade: ChatFacadeService,
    ):
        self.stream_manager = stream_manager
        self._chat_facade = chat_facade

    async def initiate_chat_stream(
        self, request: ChatRequest, message_id: str | None = None
    ) -> str:
        """Initiate a chat completion stream."""
        message_id = message_id or str(uuid4())
        if message_id and await self.stream_manager.stream_exists(message_id):
            raise ValueError("Stream with this message_id already exists")

        request = request.model_copy(
            update={
                "context": request.context.model_copy(
                    update={"correlation_id": message_id}
                )
            }
        )
        event_generator = await self._chat_facade.create_chat_event_generator(
            request=request
        )
        return await self.stream_manager.create_and_start_stream(
            event_handler=StreamingEventHandler(),
            stream_type="chat_completion",
            event_generator=event_generator,
            correlation_id=message_id,
            metadata={
                "message_count": len(request.messages),
                "thinking_enabled": request.thinking.enabled,
            },
        )

    async def get_stream_events(
        self, message_id: str
    ) -> AsyncGenerator[Event, None] | None:
        """Get Events stream for a chat completion."""
        if not await self.stream_manager.stream_exists(message_id):
            return None

        event_generator = self.stream_manager.stream_events(
            event_handler=StreamingEventHandler(), correlation_id=message_id
        )
        return await PingEventInterceptor().intercept(event_generator)

    async def cancel_stream(self, message_id: str) -> bool | None:
        """Cancel an active chat stream."""
        return await self.stream_manager.cancel_stream(message_id)

    async def clean_up_stream(self, message_id: str) -> None:
        """Cleanup resources associated with a stream."""
        await self.stream_manager.clean_up_stream(message_id)

    async def get_stream_metadata(self, message_id: str) -> StreamMetadata | None:
        """Get stream status and metadata."""
        return await self.stream_manager.get_stream_metadata(message_id)
