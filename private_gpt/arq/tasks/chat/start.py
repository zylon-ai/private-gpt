import logging
from typing import Any

from private_gpt.arq.enqueue import enqueue_job
from private_gpt.arq.tasks import arq_task
from private_gpt.arq.tasks.chat.settings import START_CHAT_TASK_NAME, get_queue_name
from private_gpt.di import get_global_injector
from private_gpt.server.chat.chat_service import ChatService
from private_gpt.settings.settings import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if settings().server.debug_mode else logging.INFO)


async def enqueue_start_chat_job(
    *,
    request_data: dict[str, Any],
    correlation_id: str,
    stream_type: str,
    metadata: dict[str, Any],
    job_id: str | None = None,
) -> None:
    logger.debug(
        "Dispatching chat start correlation_id=%s message_id=%s job_id=%s",
        correlation_id,
        correlation_id,
        job_id or f"{correlation_id}:start",
    )
    await enqueue_job(
        task_name=START_CHAT_TASK_NAME,
        queue_name=get_queue_name(settings()),
        args=(request_data, correlation_id, stream_type, metadata),
        job_id=job_id or f"{correlation_id}:start",
        correlation_id=correlation_id,
    )


@arq_task(name=START_CHAT_TASK_NAME)
async def start_chat_job(
    ctx: dict[Any, Any],
    request_data: dict[str, Any],
    correlation_id: str,
    stream_type: str,
    metadata: dict[str, Any],
) -> None:
    logger.info(
        "Chat start started correlation_id=%s message_id=%s",
        correlation_id,
        correlation_id,
    )
    injector = ctx.get("injector") or get_global_injector(
        allow_to_generate_new_injectors=True
    )
    try:
        await (
            injector.get(ChatService)
            .build_async_engine()
            .execute_scheduled_start(
                execution_id=correlation_id,
                request_data=request_data,
                stream_type=stream_type,
                metadata=metadata,
            )
        )
    except Exception:
        logger.exception(
            "Chat start failed correlation_id=%s message_id=%s",
            correlation_id,
            correlation_id,
        )
        raise
    logger.info(
        "Chat start finished correlation_id=%s message_id=%s",
        correlation_id,
        correlation_id,
    )
