from collections.abc import AsyncGenerator
from typing import Any

from injector import inject, singleton

from private_gpt.components.streaming.providers.models import (
    StreamMetadata,
    StreamStatus,
)
from private_gpt.components.streaming.stream.stream_manager import StreamManager
from private_gpt.events.event_serializer import StreamingEventHandler
from private_gpt.events.models import Event
from private_gpt.server.chat.chat_facade import ChatFacadeService
from private_gpt.server.chat.chat_models import ChatBody
from private_gpt.server.chat.chat_request_mapper import ChatRequestMapper
from private_gpt.settings.settings import settings as _settings


@singleton
class ChatAsyncService:
    """Service for initiating and observing asynchronous chat streams.

    Accepts only :class:`ChatBody` (the API-layer model). The mapping to the
    internal :class:`ChatRequest` (the business-layer model) happens inside
    this service via :class:`ChatRequestMapper`, never at the router level.
    """

    @inject
    def __init__(
        self,
        chat_facade: ChatFacadeService,
        stream_manager: StreamManager,
        chat_request_mapper: ChatRequestMapper,
    ):
        self._chat_facade = chat_facade
        self.stream_manager = stream_manager
        self._chat_request_mapper = chat_request_mapper

    async def initiate_chat_stream(
        self,
        body: ChatBody,
        message_id: str | None = None,
    ) -> str:
        """Initiate a chat completion stream from a :class:`ChatBody`.

        When ``chat.use_chat_worker`` is enabled, the chat loop is dispatched
        to a long-lived Celery worker (queue ``chat``) and the API becomes a
        pure Redis → SSE proxy. The Redis stream is created on the API side
        so the client can start reading immediately; the worker pushes events
        to that same stream.

        When the flag is off, the chat loop runs as an asyncio task on the
        API event loop (today's behaviour).

        Raises:
            ValueError: If message_id already exists.
        """
        if message_id and await self.stream_manager.stream_exists(message_id):
            raise ValueError("Stream with this message_id already exists")

        if _settings().chat.use_chat_worker:
            return await self._initiate_chat_stream_worker(
                body=body,
                message_id=message_id,
            )

        # Default in-process path: map ChatBody → ChatRequest internally.
        request = await self._chat_request_mapper.create_request_from_body(body)
        event_generator = await self._chat_facade.create_chat_event_generator(
            request=request
        )
        message_id = await self.stream_manager.create_and_start_stream(
            event_handler=StreamingEventHandler(),
            stream_type="chat_completion",
            event_generator=event_generator,
            correlation_id=message_id,
            metadata={
                "message_count": len(body.messages),
            },
        )

        return message_id

    async def _initiate_chat_stream_worker(
        self,
        body: ChatBody,
        message_id: str | None,
    ) -> str:
        """Dispatch the chat loop to the Celery ``chat`` queue.

        Only the raw :class:`ChatBody` (pure Pydantic, picklable) crosses the
        process boundary. The worker re-maps it to a :class:`ChatRequest`
        using its own warm injector so tool implementations and output schemas
        are built fresh in-process.
        """
        from private_gpt.celery.celery import celery_app

        stream_service = self.stream_manager.stream_service

        correlation_id = await stream_service.create_stream(
            stream_type="chat_completion",
            correlation_id=message_id,
            metadata={
                "message_count": len(body.messages),
            },
        )

        metadata: dict[str, Any] = {
            "message_count": len(body.messages),
        }

        body_data = body.model_dump(mode="json")
        try:
            # Use the correlation_id as the Celery task id so revocation is trivial.
            celery_app.send_task(
                "private_gpt.chat.run",
                args=[body_data, correlation_id, "chat_completion", metadata],
                queue="chat",
                task_id=correlation_id,
            )
        except Exception as e:
            await stream_service.update_stream_status(
                correlation_id,
                StreamStatus.ERROR,
                error_message=str(e),
                metadata=metadata,
            )
            raise

        return correlation_id

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
        """Cancel an active chat stream.

        Signals cancellation through two channels:
        1. The stream-service cancel flag (consumed by the worker's
           ``StreamProcessor.process_stream`` poll loop).
        2. ``revoke_task`` (Celery control-plane backstop).
        3. ``StreamManager.cancel_stream`` (in-process asyncio task cancel).
        """
        from private_gpt.celery.celery import celery_app
        from private_gpt.celery.task_helper import revoke_task

        await self.stream_manager.stream_service.set_cancel_flag(message_id)
        if _settings().chat.use_chat_worker:
            await self.stream_manager.stream_service.update_stream_status(
                message_id,
                StreamStatus.CANCELLED,
            )
        revoke_task(celery_app, message_id)

        return await self.stream_manager.cancel_stream(message_id)

    async def clean_up_stream(self, message_id: str) -> None:
        """Cleanup resources associated with a stream."""
        await self.stream_manager.clean_up_stream(message_id)

    async def get_stream_metadata(self, message_id: str) -> StreamMetadata | None:
        """Get stream status and metadata."""
        return await self.stream_manager.get_stream_metadata(message_id)
