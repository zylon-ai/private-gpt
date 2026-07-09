import contextlib
from typing import Any

from arq.worker import func

from private_gpt.arq.settings import CHAT_TASK_NAME
from private_gpt.components.streaming.providers.models import StreamStatus
from private_gpt.di import (
    clean_global_injector,
    get_global_injector,
    set_global_injector,
)
from private_gpt.eager_loading import warm
from private_gpt.events.event_serializer import StreamingEventHandler
from private_gpt.initialize import initialize_globals, initialize_observability
from private_gpt.server.chat.chat_models import ChatBody
from private_gpt.server.chat.chat_request_mapper import ChatRequestMapper
from private_gpt.server.chat.chat_service import ChatService
from private_gpt.settings.settings import settings


async def startup(ctx: dict[Any, Any]) -> None:
    current_settings = settings()
    initialize_globals()
    initialize_observability(current_settings)
    injector = get_global_injector(allow_to_generate_new_injectors=True)
    set_global_injector(injector)
    warm(injector, profile="chat")
    ctx["injector"] = injector


async def shutdown(ctx: dict[Any, Any]) -> None:
    del ctx
    clean_global_injector()


async def run_chat_job(
    ctx: dict[Any, Any],
    body: dict[str, Any],
    correlation_id: str,
    stream_type: str,
    metadata: dict[str, Any],
) -> None:
    injector = ctx.get("injector") or get_global_injector(
        allow_to_generate_new_injectors=True
    )
    chat_request_mapper = injector.get(ChatRequestMapper)
    chat_service = injector.get(ChatService)

    from private_gpt.components.streaming.stream.stream_processor import StreamProcessor

    stream_processor = injector.get(StreamProcessor)
    event_handler = StreamingEventHandler()

    try:
        chat_body = ChatBody.model_validate(body)
        request = await chat_request_mapper.create_request_from_body(chat_body)
        completion_gen = await chat_service.stream_chat(request)
    except Exception as exc:
        error_event = event_handler.error_event(correlation_id, exc)
        event_data = event_handler.serialize(error_event)
        await stream_processor.stream_service.push_event(
            correlation_id=correlation_id,
            event_data=event_data,
        )
        await stream_processor.stream_service.update_stream_status(
            correlation_id,
            StreamStatus.ERROR,
            error_message=str(exc),
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


async def on_job_end(ctx: dict[Any, Any]) -> None:
    del ctx
    with contextlib.suppress(Exception):
        injector = get_global_injector(allow_to_generate_new_injectors=True)
        set_global_injector(injector)


functions = [func(run_chat_job, name=CHAT_TASK_NAME, max_tries=1)]
