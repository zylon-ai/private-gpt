from collections.abc import AsyncGenerator

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import (
    ChatRequest,
)
from private_gpt.components.streaming.stream.stream_manager import StreamManager
from private_gpt.events.models import Event
from private_gpt.server.chat.chat_service import ChatService


@singleton
class ChatFacadeService:
    _chat_service: ChatService
    _stream_manager: StreamManager

    @inject
    def __init__(
        self,
        chat_service: ChatService,
        stream_manager: StreamManager,
    ) -> None:
        self._chat_service = chat_service
        self._stream_manager = stream_manager

    async def create_chat_event_generator(
        self, request: ChatRequest
    ) -> AsyncGenerator[Event, None]:
        """Create the chat event generator."""

        async def coro() -> AsyncGenerator[Event, None]:
            completion_gen = await self._chat_service.stream_chat(request)
            event_generator = completion_gen.events
            try:
                async for event in event_generator:
                    yield event
            finally:
                await event_generator.aclose()

        return coro()

    async def cancel(self, correlation_id: str) -> bool:
        return await self._chat_service.cancel(correlation_id)
