import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from private_gpt.arq.enqueue import abort_job, enqueue_job
from private_gpt.arq.tasks import arq_task
from private_gpt.arq.tasks.chat.settings import (
    RESUME_ITERATION_TASK_NAME,
    TOOL_RESUME_TASK_NAME,
    TOOL_TIMEOUT_TASK_NAME,
    get_queue_name,
)
from private_gpt.components.engines.chat.async_chat_engine import AsyncChatEngine
from private_gpt.components.tools.remote_execution import ToolExecutionResponse
from private_gpt.components.tools.tool_scheduler import ToolSchedulerFactory
from private_gpt.di import get_global_injector
from private_gpt.events.models import TextBlock
from private_gpt.server.chat.chat_service import ChatService
from private_gpt.settings.settings import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if settings().server.debug_mode else logging.INFO)


async def enqueue_resume_iteration_job(
    *, correlation_id: str, checkpoint_id: str, job_id: str
) -> None:
    logger.debug(
        "Dispatching chat resume correlation_id=%s message_id=%s job_id=%s",
        correlation_id,
        correlation_id,
        job_id,
    )
    await enqueue_job(
        task_name=RESUME_ITERATION_TASK_NAME,
        queue_name=get_queue_name(settings()),
        args=(correlation_id, checkpoint_id),
        job_id=job_id,
        correlation_id=correlation_id,
    )


async def enqueue_tool_timeout_job(
    *,
    correlation_id: str,
    checkpoint_id: str,
    tool_id: str,
    tool_name: str,
    task_id: str,
    delay_seconds: int,
) -> None:
    job_id = f"{correlation_id}:tool-timeout:{checkpoint_id}:{tool_id}"
    logger.debug(
        "Dispatching tool timeout correlation_id=%s message_id=%s "
        "checkpoint_id=%s tool_id=%s job_id=%s delay_seconds=%s",
        correlation_id,
        correlation_id,
        checkpoint_id,
        tool_id,
        job_id,
        delay_seconds,
    )
    await enqueue_job(
        task_name=TOOL_TIMEOUT_TASK_NAME,
        queue_name=get_queue_name(settings()),
        args=(correlation_id, tool_id, tool_name, task_id, delay_seconds),
        job_id=job_id,
        correlation_id=correlation_id,
        defer_seconds=delay_seconds,
    )


async def abort_tool_timeout_job(
    *, correlation_id: str, checkpoint_id: str, tool_id: str
) -> bool:
    job_id = f"{correlation_id}:tool-timeout:{checkpoint_id}:{tool_id}"
    logger.debug(
        "Aborting obsolete tool timeout correlation_id=%s message_id=%s "
        "checkpoint_id=%s tool_id=%s job_id=%s",
        correlation_id,
        correlation_id,
        checkpoint_id,
        tool_id,
        job_id,
    )
    return await abort_job(job_id=job_id, queue_name=get_queue_name(settings()))


async def enqueue_tool_resume_job(
    *, correlation_id: str, tool_id: str, result: dict[str, Any]
) -> bool:
    job_id = f"{correlation_id}:tool-result:{tool_id}"
    logger.debug(
        "Dispatching tool callback correlation_id=%s message_id=%s tool_id=%s "
        "job_id=%s",
        correlation_id,
        correlation_id,
        tool_id,
        job_id,
    )
    return await enqueue_job(
        task_name=TOOL_RESUME_TASK_NAME,
        queue_name=get_queue_name(settings()),
        args=(correlation_id, tool_id, result),
        job_id=job_id,
        correlation_id=correlation_id,
    )


def _timeout_response(
    *, tool_id: str, tool_name: str, delay_seconds: int
) -> ToolExecutionResponse:
    from llama_index.core.base.llms.types import ChatMessage

    message = (
        f"Tool execution timed out after {delay_seconds} seconds and was cancelled "
        "before returning a result."
    )
    return ToolExecutionResponse(
        tool_name=tool_name,
        tool_id=tool_id,
        result_content=[TextBlock(text=message)],
        is_error=True,
        tool_message=ChatMessage(
            role="tool",
            content=message,
            additional_kwargs={
                "tool_call_id": tool_id,
                "tool_call_name": tool_name,
                "raw_output": message,
            },
        ),
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
        function_name=TOOL_RESUME_TASK_NAME,
        operation=lambda: _engine(ctx).record_callback(
            execution_id=correlation_id, tool_id=tool_id, result=result
        ),
    )


@arq_task(name=RESUME_ITERATION_TASK_NAME)
async def resume_iteration_job(
    ctx: dict[Any, Any], correlation_id: str, checkpoint_id: str
) -> None:
    await _run_logged(
        ctx,
        action="resume",
        correlation_id=correlation_id,
        function_name=RESUME_ITERATION_TASK_NAME,
        operation=lambda: _engine(ctx).execute_scheduled_resume(
            execution_id=correlation_id,
            checkpoint_id=checkpoint_id,
        ),
    )


@arq_task(name=TOOL_TIMEOUT_TASK_NAME)
async def timeout_tool_job(
    ctx: dict[Any, Any],
    correlation_id: str,
    tool_id: str,
    tool_name: str,
    task_id: str,
    delay_seconds: int,
) -> None:
    del ctx
    response = _timeout_response(
        tool_id=tool_id,
        tool_name=tool_name,
        delay_seconds=delay_seconds,
    )
    accepted = await enqueue_tool_resume_job(
        correlation_id=correlation_id,
        tool_id=tool_id,
        result=response.model_dump(mode="json"),
    )
    if accepted:
        injector = get_global_injector(allow_to_generate_new_injectors=True)
        await injector.get(ToolSchedulerFactory).get().cancel_task(task_id)


async def _run_logged(
    ctx: dict[Any, Any],
    *,
    action: str,
    correlation_id: str,
    operation: Callable[[], Awaitable[None]],
    tool_id: str | None = None,
    function_name: str = "unknown",
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
    except asyncio.CancelledError:
        logger.warning(
            "Chat action cancelled action=%s correlation_id=%s message_id=%s tool_id=%s",
            action,
            correlation_id,
            correlation_id,
            tool_id,
        )
        raise
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
