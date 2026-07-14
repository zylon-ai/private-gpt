from typing import Any

from private_gpt.arq.settings import (
    CHAT_TIMEOUT_TASK_NAME,
    RESUME_ITERATION_TASK_NAME,
    TOOL_RESUME_TASK_NAME,
)
from private_gpt.arq.tasks import arq_task
from private_gpt.components.engines.chat.async_chat_engine import AsyncChatEngine
from private_gpt.di import get_global_injector
from private_gpt.server.chat.chat_service import ChatService


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
