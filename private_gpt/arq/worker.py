import contextlib
from typing import Any

import nest_asyncio  # type: ignore
from arq.worker import func

from private_gpt.arq.iteration_state import IterationContext, IterationStateService
from private_gpt.arq.settings import CHAT_TASK_NAME, TOOL_DONE_TASK_NAME
from private_gpt.components.chat.models.chat_config_models import ResolvedChatRequest
from private_gpt.components.engines.chat_loop.models.chat_loop_state import (
    ChatLoopStatus,
)
from private_gpt.components.engines.chat_loop.models.execution_hooks import (
    ToolExecutionHook,
)
from private_gpt.components.streaming.providers.models import StreamStatus
from private_gpt.components.tools.remote_execution import (
    ToolExecutionRequest,
    ToolExecutionResponse,
)
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
    nest_asyncio.apply()
    injector = get_global_injector(allow_to_generate_new_injectors=True)
    set_global_injector(injector)
    warm(injector, profile="chat")
    ctx["injector"] = injector


async def shutdown(ctx: dict[Any, Any]) -> None:
    del ctx
    await clean_global_injector()


async def run_chat_job(
    ctx: dict[Any, Any],
    body: dict[str, Any],
    correlation_id: str,
    stream_type: str,
    metadata: dict[str, Any],
    request_data: dict[str, Any] | None = None,
) -> None:
    injector = ctx.get("injector") or get_global_injector(
        allow_to_generate_new_injectors=True
    )
    chat_request_mapper = injector.get(ChatRequestMapper)
    chat_service = injector.get(ChatService)

    from private_gpt.components.streaming.stream.stream_processor import StreamProcessor

    stream_processor = injector.get(StreamProcessor)
    iteration_state_service = injector.get(IterationStateService)
    event_handler = StreamingEventHandler()

    try:
        if request_data is None:
            chat_body = ChatBody.model_validate(body)
            request = await chat_request_mapper.create_request_from_body(chat_body)
        else:
            request = ResolvedChatRequest.model_validate(request_data)
        completion_gen = await chat_service.stream_chat(
            request,
            hooks=[
                ToolExecutionHook(
                    callable_path="private_gpt.arq.worker.resume_chat_callback",
                )
            ],
        )
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
        mark_completed=False,
    )

    final_state = (
        await completion_gen.final_state_task
        if completion_gen.final_state_task is not None
        else None
    )
    if final_state and final_state.output.status == ChatLoopStatus.WAITING:
        await iteration_state_service.save(
            IterationContext(
                correlation_id=correlation_id,
                body=body,
                request_data=final_state.input.request.model_dump(mode="json"),
                pending_async_tools=final_state.output.pending_async_tools,
                stream_type=stream_type,
                metadata=metadata,
                iteration=final_state.runtime.iteration,
            )
        )
        return

    await stream_processor.stream_service.update_stream_status(
        correlation_id,
        StreamStatus.COMPLETED,
        metadata=metadata,
    )


async def tool_done_job(
    ctx: dict[Any, Any],
    correlation_id: str,
    tool_id: str,
    result: dict[str, Any],
) -> None:
    injector = ctx.get("injector") or get_global_injector(
        allow_to_generate_new_injectors=True
    )
    iteration_state_service = injector.get(IterationStateService)
    from private_gpt.components.streaming.stream.stream_processor import StreamProcessor

    stream_processor = injector.get(StreamProcessor)
    saved = await iteration_state_service.load(correlation_id)
    if saved is None:
        return
    if await stream_processor.stream_service.is_cancelled(correlation_id):
        await iteration_state_service.cleanup(correlation_id)
        return

    all_results = await iteration_state_service.record_result(
        correlation_id, tool_id, result
    )
    if all_results is None:
        return
    if not await iteration_state_service.claim_resume(correlation_id):
        return

    request_data = saved.request_data
    messages = request_data.get("messages", [])
    for response in all_results.values():
        validated = (
            response
            if isinstance(response, ToolExecutionResponse)
            else ToolExecutionResponse.model_validate(response)
        )
        messages.append(validated.tool_message.model_dump(mode="json"))
    request_data["messages"] = messages

    from private_gpt.arq.enqueue import enqueue_chat_job

    await enqueue_chat_job(
        body=saved.body,
        correlation_id=correlation_id,
        stream_type=saved.stream_type,
        metadata=saved.metadata,
        request_data=request_data,
        job_id=f"{correlation_id}:iter:{saved.iteration + 1}",
    )
    await iteration_state_service.cleanup(correlation_id)


async def resume_chat_callback(
    *,
    request: ToolExecutionRequest,
    response: ToolExecutionResponse,
) -> None:
    correlation_id = request.context.get("correlation_id")
    if not correlation_id or not request.tool_id:
        return

    from private_gpt.arq.enqueue import enqueue_tool_done_job

    await enqueue_tool_done_job(
        correlation_id=correlation_id,
        tool_id=request.tool_id,
        result=response.model_dump(mode="json"),
    )


async def on_job_end(ctx: dict[Any, Any]) -> None:
    del ctx
    with contextlib.suppress(Exception):
        injector = get_global_injector(allow_to_generate_new_injectors=True)
        set_global_injector(injector)


functions = [
    func(run_chat_job, name=CHAT_TASK_NAME, max_tries=1),
    func(tool_done_job, name=TOOL_DONE_TASK_NAME, max_tries=1),
]
