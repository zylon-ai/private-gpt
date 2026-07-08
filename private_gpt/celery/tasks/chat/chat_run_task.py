"""Celery task that runs the entire chat loop in a long-lived worker process.

When ``chat.use_chat_worker`` is enabled, the FastAPI server dispatches chat
requests here instead of running the chat loop on its own event loop. The
worker pushes events to the same Redis stream the API reads from, so the
API becomes a pure Redis → SSE proxy and its event loop never contends with
CPU-bound work (LLM calls, tools, semantic search, retrieval, tokenization).
"""
import logging
from typing import Any

from private_gpt.celery.base import ChatBackgroundTask
from private_gpt.celery.celery import celery_app
from private_gpt.components.streaming.providers.models import StreamStatus
from private_gpt.di import get_global_injector
from private_gpt.events.event_serializer import StreamingEventHandler
from private_gpt.server.chat.chat_models import ChatBody
from private_gpt.server.chat.chat_request_mapper import ChatRequestMapper
from private_gpt.settings.settings import settings

logger = logging.getLogger(__name__)


@celery_app.task(
    name="private_gpt.chat.run",
    base=ChatBackgroundTask,
)
async def chat_run_task(
    body: dict[str, Any],
    correlation_id: str,
    stream_type: str,
    metadata: dict[str, Any],
) -> None:
    """Run a chat completion loop and push events to the stream.

    Args:
        body: JSON-safe :class:`ChatBody` data received by the API. It is re-mapped
            to a :class:`ChatRequest` inside the worker so that tool
            implementations and output schemas are built from the worker's
            warm dependency injector (no closures or dynamic classes cross
            the process boundary).
        correlation_id: The Redis stream correlation ID already created by the
            API via ``StreamService.create_stream``.
        stream_type: The stream type (e.g. ``"chat_completion"``).
        metadata: Metadata to attach to the stream.
    """
    injector = get_global_injector()
    chat_request_mapper = injector.get(ChatRequestMapper)

    from private_gpt.components.streaming.stream.stream_processor import (
        StreamProcessor,
    )
    from private_gpt.server.chat.chat_service import ChatService

    chat_service = injector.get(ChatService)
    stream_processor = injector.get(StreamProcessor)
    event_handler = StreamingEventHandler()

    try:
        chat_body = ChatBody.model_validate(body)
        request = await chat_request_mapper.create_request_from_body(chat_body)
        completion_gen = await chat_service.stream_chat(request)
    except Exception as e:
        error_event = event_handler.error_event(correlation_id, e)
        event_data = event_handler.serialize(error_event)
        await stream_processor.stream_service.push_event(
            correlation_id=correlation_id,
            event_data=event_data,
        )
        await stream_processor.stream_service.update_stream_status(
            correlation_id,
            StreamStatus.ERROR,
            error_message=str(e),
            metadata=metadata,
        )
        raise

    await stream_processor.process_stream(
        correlation_id=correlation_id,
        stream_type=stream_type,
        event_generator=completion_gen.events,  # type: ignore[arg-type]
        event_handler=event_handler,
        metadata=metadata,
    )

    logger.debug(
        "chat_run_task completed for correlation_id=%s (debug=%s)",
        correlation_id,
        settings().server.debug_mode,
    )
