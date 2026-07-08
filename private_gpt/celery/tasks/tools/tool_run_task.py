"""Celery task that executes a single tool call on a dedicated tools worker."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from private_gpt.celery.base import StatefulBackgroundTask
from private_gpt.celery.celery import celery_app
from private_gpt.components.tools.remote_execution import (
    ToolExecutionResponse,
    execute_tool_request,
)
from private_gpt.events.models import TextBlock

if TYPE_CHECKING:
    from private_gpt.components.tools.remote_execution import ToolExecutionRequest

logger = logging.getLogger(__name__)


@celery_app.task(
    name="private_gpt.tools.run",
    base=StatefulBackgroundTask,
)
async def tool_run_task(request: ToolExecutionRequest) -> dict[str, Any]:
    try:
        response = await execute_tool_request(request)
        return response.model_dump(mode="json")
    except Exception as exc:
        logger.exception("Tool '%s' execution failed", request.tool_name)
        return ToolExecutionResponse(
            tool_name=request.tool_name,
            tool_id=request.tool_id,
            result_content=[TextBlock(text=str(exc))],
            is_error=True,
            tool_message=request_error_message(request, str(exc)),
        ).model_dump(mode="json")


def request_error_message(
    request: ToolExecutionRequest,
    message: str,
) -> Any:
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
