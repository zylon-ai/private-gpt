from typing import Any

from private_gpt.arq.event_channel import RedisEventChannel
from private_gpt.arq.iteration_state import IterationStateService
from private_gpt.arq.settings import START_CHAT_TASK_NAME
from private_gpt.arq.tasks import arq_task
from private_gpt.components.engines.chat.execution.chat_step import (
    execute_chat_start_step,
)
from private_gpt.components.engines.chat.models.chat_state import ChatStatus
from private_gpt.components.engines.chat.models.execution_hooks import (
    ExecutionHooks,
    ToolExecutionHook,
)
from private_gpt.components.engines.chat.schedulers.iteration_scheduler import (
    IterationScheduler,
)
from private_gpt.di import get_global_injector
from private_gpt.events.event_serializer import StreamingEventHandler
from private_gpt.server.chat.chat_service import ChatService

_ARQ_HOOKS = ExecutionHooks(
    tool_result=[
        ToolExecutionHook(
            callable_path="private_gpt.arq.tasks.chat.callback.resume_chat_callback",
        )
    ]
)


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
    chat_service = injector.get(ChatService)
    iteration_state_service = injector.get(IterationStateService)
    event_handler = StreamingEventHandler()

    channel = RedisEventChannel(iteration_state_service, correlation_id, event_handler)
    try:
        state = await execute_chat_start_step(
            chat_service=chat_service,
            request_data=request_data,
            hooks=_ARQ_HOOKS,
            channel=channel,
        )
    except Exception as exc:
        await iteration_state_service.append_event(
            correlation_id,
            event_handler.serialize(event_handler.error_event(correlation_id, exc)),
        )
        await channel.close()
        raise

    if state.output.status == ChatStatus.WAITING:
        scheduler = IterationScheduler(
            iteration_state_service=iteration_state_service,
            correlation_id=correlation_id,
            request_data=request_data,
            stream_type=stream_type,
            metadata=metadata,
        )
        await scheduler.on_waiting(state)
    else:
        await channel.close()
