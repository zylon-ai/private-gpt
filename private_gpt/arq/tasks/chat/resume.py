import logging
from collections.abc import Awaitable, Callable
from typing import Any

from private_gpt.arq.enqueue import enqueue_job
from private_gpt.arq.tasks import arq_task
from private_gpt.arq.tasks.chat.settings import (
    CHAT_TIMEOUT_TASK_NAME,
    RESUME_ITERATION_TASK_NAME,
    TOOL_RESUME_TASK_NAME,
    get_queue_name,
)
from private_gpt.components.engines.chat.async_chat_engine import AsyncChatEngine
from private_gpt.di import get_global_injector
from private_gpt.server.chat.chat_service import ChatService
from private_gpt.settings.settings import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if settings().server.debug_mode else logging.INFO)


async def enqueue_resume_iteration_job(
    *, correlation_id: str, job_id: str | None = None
) -> None:
    resolved_job_id = job_id or f"{correlation_id}:resume"
    logger.debug(
        "Dispatching chat resume correlation_id=%s message_id=%s job_id=%s",
        correlation_id,
        correlation_id,
        resolved_job_id,
    )
    await enqueue_job(
        task_name=RESUME_ITERATION_TASK_NAME,
        queue_name=get_queue_name(settings()),
        args=(correlation_id,),
        job_id=resolved_job_id,
        correlation_id=correlation_id,
    )


async def enqueue_chat_timeout_job(
    *,
    correlation_id: str,
    checkpoint_id: str,
    delay_seconds: int,
    job_id: str,
) -> None:
    logger.debug(
        "Dispatching chat timeout correlation_id=%s message_id=%s "
        "checkpoint_id=%s job_id=%s delay_seconds=%s",
        correlation_id,
        correlation_id,
        checkpoint_id,
        job_id,
        delay_seconds,
    )
    await enqueue_job(
        task_name=CHAT_TIMEOUT_TASK_NAME,
        queue_name=get_queue_name(settings()),
        args=(correlation_id, checkpoint_id),
        job_id=job_id,
        correlation_id=correlation_id,
        defer_seconds=delay_seconds,
    )


async def enqueue_tool_resume_job(
    *, correlation_id: str, tool_id: str, result: dict[str, Any]
) -> None:
    logger.debug(
        "Dispatching tool callback correlation_id=%s message_id=%s tool_id=%s",
        correlation_id,
        correlation_id,
        tool_id,
    )
    await enqueue_job(
        task_name=TOOL_RESUME_TASK_NAME,
        queue_name=get_queue_name(settings()),
        args=(correlation_id, tool_id, result),
        correlation_id=correlation_id,
    )


def _engine(ctx: dict[Any, Any]) -> AsyncChatEngine:
    injector = ctx.get("injector") or get_global_injector(
        allow_to_generate_new_injectors=True
    )
    return injector.get(ChatService).build_async_engine()


@arq_task(name=TOOL_RESUME_TASK_NAME)
async def tool_resume_job(
    ctx: dict[Any, Any],
    correlation_id: str,
    tool_id: str,
    result: dict[str, Any],
) -> None:
    await _run_logged(
        ctx,
        action="tool_callback",
        correlation_id=correlation_id,
        tool_id=tool_id,
        operation=lambda: _engine(ctx).record_callback(
            execution_id=correlation_id, tool_id=tool_id, result=result
        ),
    )


@arq_task(name=RESUME_ITERATION_TASK_NAME)
async def resume_iteration_job(ctx: dict[Any, Any], correlation_id: str) -> None:
    await _run_logged(
        ctx,
        action="resume",
        correlation_id=correlation_id,
        operation=lambda: _engine(ctx).execute_scheduled_resume(
            execution_id=correlation_id
        ),
    )


@arq_task(name=CHAT_TIMEOUT_TASK_NAME)
async def timeout_chat_job(
    ctx: dict[Any, Any], correlation_id: str, checkpoint_id: str
) -> None:
    await _run_logged(
        ctx,
        action="timeout",
        correlation_id=correlation_id,
        operation=lambda: _engine(ctx).execute_timeout(
            execution_id=correlation_id, checkpoint_id=checkpoint_id
        ),
    )


async def _run_logged(
    ctx: dict[Any, Any],
    *,
    action: str,
    correlation_id: str,
    operation: Callable[[], Awaitable[None]],
    tool_id: str | None = None,
) -> None:
    logger.info(
        "Chat action started action=%s correlation_id=%s message_id=%s tool_id=%s",
        action,
        correlation_id,
        correlation_id,
        tool_id,
    )
    try:
        await operation()
    except Exception:
        logger.exception(
            "Chat action failed action=%s correlation_id=%s message_id=%s tool_id=%s",
            action,
            correlation_id,
            correlation_id,
            tool_id,
        )
        raise
    logger.info(
        "Chat action finished action=%s correlation_id=%s message_id=%s tool_id=%s",
        action,
        correlation_id,
        correlation_id,
        tool_id,
    )
