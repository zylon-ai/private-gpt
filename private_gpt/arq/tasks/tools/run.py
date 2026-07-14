import logging
from typing import Any

from private_gpt.arq.settings import TOOL_RUN_TASK_NAME
from private_gpt.arq.tasks import arq_task
from private_gpt.components.tools.remote_execution import (
    ToolExecutionRequest,
    ToolExecutionResponse,
    execute_tool_request,
)
from private_gpt.events.models import TextBlock

logger = logging.getLogger(__name__)


@arq_task(name=TOOL_RUN_TASK_NAME)
async def run_tool_job(ctx: dict[Any, Any], request_data: dict[str, Any]) -> None:
    del ctx
    request = ToolExecutionRequest.model_validate(request_data)
    try:
        response = await execute_tool_request(request)
    except Exception as exc:
        logger.exception("Tool '%s' execution failed", request.tool_name)
        response = ToolExecutionResponse(
            tool_name=request.tool_name,
            tool_id=request.tool_id,
            result_content=[TextBlock(text=str(exc))],
            is_error=True,
            tool_message=_request_error_message(request, str(exc)),
        )

    from private_gpt.components.tools.remote_execution import invoke_execution_hook

    for hook in request.hooks.tool_result:
        await invoke_execution_hook(hook, request, response)


def _request_error_message(request: ToolExecutionRequest, message: str) -> Any:
    from llama_index.core.base.llms.types import ChatMessage

    return ChatMessage(
        role="tool",
        content=message,
        additional_kwargs={
            "tool_call_id": request.tool_id,
            "tool_call_name": request.tool_name,
            "tool_call_args": request.tool_kwargs,
            "raw_output": message,
        },
    )
