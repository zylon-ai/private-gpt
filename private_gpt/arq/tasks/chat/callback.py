from private_gpt.components.tools.remote_execution import (
    ToolExecutionRequest,
    ToolExecutionResponse,
)


async def resume_chat_callback(
    *,
    request: ToolExecutionRequest,
    response: ToolExecutionResponse,
) -> None:
    correlation_id = request.context.get("correlation_id")
    if not correlation_id or not request.tool_id:
        return

    from private_gpt.components.engines.chat.execution_scheduler import (
        ChatExecutionSchedulerFactory,
    )
    from private_gpt.di import get_global_injector

    injector = get_global_injector(allow_to_generate_new_injectors=True)
    scheduler = injector.get(ChatExecutionSchedulerFactory).get()
    await scheduler.callback(
        execution_id=correlation_id,
        tool_id=request.tool_id,
        result=response.model_dump(mode="json"),
    )
