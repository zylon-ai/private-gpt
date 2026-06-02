import asyncio
import logging
from collections.abc import AsyncGenerator

from injector import inject, singleton
from starlette.requests import Request
from starlette.responses import StreamingResponse

from private_gpt.components.chat.models.chat_config_models import ChatRequest
from private_gpt.events.event_folding import fold
from private_gpt.events.models import Event, FatalError, Message
from private_gpt.events.utils import to_message, to_sse_stream
from private_gpt.server.chat_async.chat_async_service import ChatAsyncService

logger = logging.getLogger(__name__)


@singleton
class ChatAsyncFacadeService:

    _chat_async_service: ChatAsyncService

    @inject
    def __init__(
        self,
        chat_async_service: ChatAsyncService,
    ) -> None:
        self._chat_async_service = chat_async_service

    async def chat(
        self, http_request: Request, request: ChatRequest, message_id: str | None = None
    ) -> Message | FatalError | StreamingResponse:
        """Handle chat with proper cancellation support for FastAPI.

        When FastAPI request is cancelled (client disconnects), this ensures:
        1. Stream is properly cancelled in the StreamManager
        2. Resources are cleaned up
        3. CancelledError is re-raised to maintain asyncio semantics
        """
        message_id = await self._chat_async_service.initiate_chat_stream(
            request=request, message_id=message_id
        )
        event_generator = await self._chat_async_service.get_stream_events(
            message_id=message_id,
        )
        if event_generator is None:
            raise ValueError(f"No event generator found for message_id: {message_id}")

        cancellable_generator = self._cancellable_stream_generator(
            http_request, event_generator, message_id
        )
        if request.stream:
            sse_stream = to_sse_stream(cancellable_generator)
            return StreamingResponse(
                sse_stream,
                media_type="text/event-stream",
            )
        else:
            chat_response = await fold(cancellable_generator)
            if chat_response.exception:
                raise chat_response.exception
            return to_message(
                content=chat_response.content,
                exception=chat_response.exception,
                stop_reason=chat_response.stop_reason,
                usage=chat_response.usage,
            )

    async def _cancellable_stream_generator(
        self,
        http_request: Request,
        event_generator: AsyncGenerator[Event, None],
        message_id: str,
    ) -> AsyncGenerator[Event, None]:
        """Wrap event generator to handle cancellation during streaming."""
        try:

            async def check_disconnection() -> None:
                if await http_request.is_disconnected():
                    logger.debug("HTTP request was disconnected, cleaning up stream")
                    raise asyncio.CancelledError("HTTP request was disconnected")

            await check_disconnection()
            async for event in event_generator:
                await check_disconnection()
                yield event

        except asyncio.CancelledError:
            logger.debug(f"Stream generator cancelled, cleaning up: {message_id}")
            try:
                await self._chat_async_service.cancel_stream(message_id)
            except Exception as cleanup_error:
                logger.warning(f"Error during stream cleanup: {cleanup_error}")
            raise
        finally:
            logger.debug(f"Stream generator completed for {message_id}")
            await self._chat_async_service.clean_up_stream(message_id)
