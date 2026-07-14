from typing import Any

from private_gpt.arq.enqueue import enqueue_job
from private_gpt.arq.settings import (
    CHAT_TIMEOUT_TASK_NAME,
    RESUME_ITERATION_TASK_NAME,
    TOOL_RESUME_TASK_NAME,
)
from private_gpt.arq.tasks import arq_task
from private_gpt.components.engines.chat.async_chat_engine import AsyncChatEngine
from private_gpt.di import get_global_injector
from private_gpt.server.chat.chat_service import ChatService


async def enqueue_resume_iteration_job(
    *, correlation_id: str, job_id: str | None = None
) -> None:
    await enqueue_job(
        task_name=RESUME_ITERATION_TASK_NAME,
        args=(correlation_id,),
        job_id=job_id or f"{correlation_id}:resume",
        correlation_id=correlation_id,
    )


async def enqueue_chat_timeout_job(
    *,
    correlation_id: str,
    checkpoint_id: str,
    delay_seconds: int,
    job_id: str,
) -> None:
    await enqueue_job(
        task_name=CHAT_TIMEOUT_TASK_NAME,
        args=(correlation_id, checkpoint_id),
        job_id=job_id,
        correlation_id=correlation_id,
        defer_seconds=delay_seconds,
    )


async def enqueue_tool_resume_job(
    *, correlation_id: str, tool_id: str, result: dict[str, Any]
) -> None:
    await enqueue_job(
        task_name=TOOL_RESUME_TASK_NAME,
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
    await _engine(ctx).record_callback(
        execution_id=correlation_id, tool_id=tool_id, result=result
    )


@arq_task(name=RESUME_ITERATION_TASK_NAME)
async def resume_iteration_job(ctx: dict[Any, Any], correlation_id: str) -> None:
    await _engine(ctx).execute_scheduled_resume(execution_id=correlation_id)


@arq_task(name=CHAT_TIMEOUT_TASK_NAME)
async def timeout_chat_job(
    ctx: dict[Any, Any], correlation_id: str, checkpoint_id: str
) -> None:
    await _engine(ctx).execute_timeout(
        execution_id=correlation_id, checkpoint_id=checkpoint_id
    )
