import logging

from private_gpt.components.tools.remote_execution import (
    ToolExecutionRequest,
    ToolExecutionResponse,
)
from private_gpt.settings.settings import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if settings().server.debug_mode else logging.INFO)


async def resume_chat_callback(
    *,
    request: ToolExecutionRequest,
    response: ToolExecutionResponse,
) -> None:
    correlation_id = request.context.get("correlation_id")
    if not correlation_id or not request.tool_id:
        logger.debug(
            "Skipping tool callback correlation_id=%s message_id=%s tool_id=%s",
            correlation_id,
            request.context.get("message_id") or correlation_id,
            request.tool_id,
        )
        return

    message_id = request.context.get("message_id") or correlation_id
    logger.debug(
        "Tool callback received correlation_id=%s message_id=%s tool_id=%s is_error=%s",
        correlation_id,
        message_id,
        request.tool_id,
        response.is_error,
    )

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
    logger.debug(
        "Tool callback dispatched correlation_id=%s message_id=%s tool_id=%s "
        "is_error=%s",
        correlation_id,
        message_id,
        request.tool_id,
        response.is_error,
    )
