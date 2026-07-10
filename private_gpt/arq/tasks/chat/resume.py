from typing import Any

from private_gpt.arq.enqueue import enqueue_resume_iteration_job
from private_gpt.arq.event_channel import RedisEventChannel
from private_gpt.arq.iteration_state import IterationStateService
from private_gpt.arq.settings import RESUME_ITERATION_TASK_NAME, TOOL_RESUME_TASK_NAME
from private_gpt.arq.tasks import arq_task
from private_gpt.components.chat.models.chat_config_models import ResolvedChatRequest
from private_gpt.components.engines.chat.models.chat_state import ChatStatus
from private_gpt.components.engines.chat.models.execution_hooks import (
    ExecutionHooks,
    ToolExecutionHook,
)
from private_gpt.components.engines.chat.schedulers.iteration_scheduler import (
    IterationScheduler,
)
from private_gpt.components.tools.remote_execution import ToolExecutionResponse
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


@arq_task(name=TOOL_RESUME_TASK_NAME)
async def tool_resume_job(
    ctx: dict[Any, Any],
    correlation_id: str,
    tool_id: str,
    result: dict[str, Any],
) -> None:
    injector = ctx.get("injector") or get_global_injector(
        allow_to_generate_new_injectors=True
    )
    iteration_state_service = injector.get(IterationStateService)

    saved = await iteration_state_service.load(correlation_id)
    if saved is None:
        return

    all_results = await iteration_state_service.record_result(
        correlation_id, tool_id, result
    )
    if all_results is None:
        return
    if not await iteration_state_service.claim_resume(correlation_id):
        return

    request_data = dict(saved.request_data)
    messages = list(request_data.get("messages", []))
    for response in all_results.values():
        validated = (
            response
            if isinstance(response, ToolExecutionResponse)
            else ToolExecutionResponse.model_validate(response)
        )
        messages.append(validated.tool_message.model_dump(mode="json"))
    request_data["messages"] = messages

    await enqueue_resume_iteration_job(
        body=saved.body,
        correlation_id=correlation_id,
        stream_type=saved.stream_type,
        metadata=saved.metadata,
        request_data=request_data,
        pause_type=saved.checkpoint,
        tool_results=[
            response.model_dump(mode="json") for response in all_results.values()
        ],
        iteration=saved.iteration,
        next_block_count=saved.next_block_count,
        job_id=f"{correlation_id}:resume:{saved.iteration}",
    )
    await iteration_state_service.cleanup(correlation_id)


@arq_task(name=RESUME_ITERATION_TASK_NAME)
async def resume_iteration_job(
    ctx: dict[Any, Any],
    body: dict[str, Any],
    correlation_id: str,
    stream_type: str,
    metadata: dict[str, Any],
    request_data: dict[str, Any],
    pause_type: str,
    tool_results: list[dict[str, Any]],
    iteration: int,
    next_block_count: int,
) -> None:
    injector = ctx.get("injector") or get_global_injector(
        allow_to_generate_new_injectors=True
    )
    chat_service = injector.get(ChatService)
    iteration_state_service = injector.get(IterationStateService)
    event_handler = StreamingEventHandler()

    channel = RedisEventChannel(iteration_state_service, correlation_id, event_handler)
    try:
        saved = await iteration_state_service.load(correlation_id)
        if saved is None:
            await iteration_state_service.append_done(correlation_id)
            return
        request = ResolvedChatRequest.model_validate(request_data)
        engine = chat_service.build_engine()
        state = await engine.resume(
            pause_type,
            request,
            iteration=iteration,
            next_block_count=next_block_count,
            hooks=_ARQ_HOOKS,
            checkpoint_payload=saved.checkpoint_payload.model_copy(
                update={
                    "tool_responses": [
                        ToolExecutionResponse.model_validate(item)
                        for item in tool_results
                    ]
                }
            ),
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
            body=body,
            stream_type=stream_type,
            metadata=metadata,
        )
        await scheduler.on_waiting(state)
    else:
        await channel.close()
