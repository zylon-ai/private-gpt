from typing import Any

from private_gpt.arq.settings import START_CHAT_TASK_NAME
from private_gpt.arq.tasks import arq_task
from private_gpt.di import get_global_injector
from private_gpt.server.chat.chat_service import ChatService


@arq_task(name=START_CHAT_TASK_NAME)
async def start_chat_job(
    ctx: dict[Any, Any],
    request_data: dict[str, Any],
    correlation_id: str,
    stream_type: str,
    metadata: dict[str, Any],
) -> None:
    injector = ctx.get("injector") or get_global_injector(
        allow_to_generate_new_injectors=True
    )
    await injector.get(ChatService).build_async_engine().execute_scheduled_start(
        execution_id=correlation_id,
        request_data=request_data,
        stream_type=stream_type,
        metadata=metadata,
    )
