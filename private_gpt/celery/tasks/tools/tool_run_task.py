"""Celery task that executes a single tool call on a dedicated tools worker."""

from __future__ import annotations

import json
import logging
from typing import Any

from celery import current_task

from private_gpt.celery.base import StatefulBackgroundTask
from private_gpt.celery.celery import celery_app
from private_gpt.components.tools.remote_execution import (
    ToolExecutionRequest,
    ToolExecutionResponse,
    execute_tool_request,
    resolve_tool_execution_interceptors,
)
from private_gpt.components.tools.tool_scheduler import ToolSchedulerFactory
from private_gpt.di import get_global_injector
from private_gpt.events.models import TextBlock
from private_gpt.settings.settings import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if settings().server.debug_mode else logging.INFO)

RESULT_FRAGMENT_LENGTH = 200


@celery_app.task(
    name="private_gpt.tools.run",
    base=StatefulBackgroundTask,
    ignore_result=True,
)
async def tool_run_task(*, request_data: dict[str, Any]) -> dict[str, Any]:
    try:
        request = ToolExecutionRequest.model_validate(request_data)
    except Exception:
        logger.exception(
            "Invalid tool execution request task_id=%s",
            current_task.request.id,
        )
        raise

    correlation_id = request.context.get("correlation_id")
    message_id = request.context.get("message_id") or correlation_id
    task_id = current_task.request.id
    logger.info(
        "Tool execution started correlation_id=%s message_id=%s tool_id=%s",
        correlation_id,
        message_id,
        request.tool_id,
    )
    try:
        response = await execute_tool_request(
            request,
            interceptors=resolve_tool_execution_interceptors(request.interceptor_paths),
        )
    except Exception as exc:
        logger.exception(
            "Tool execution raised an exception correlation_id=%s message_id=%s "
            "tool_id=%s",
            correlation_id,
            message_id,
            request.tool_id,
        )
        response = ToolExecutionResponse(
            tool_name=request.tool_name,
            tool_id=request.tool_id,
            result_content=[TextBlock(text=str(exc))],
            is_error=True,
            tool_message=request_error_message(request, str(exc)),
        )
    else:
        logger.debug(
            "Tool execution completed correlation_id=%s "
            "message_id=%s task_id=%s tool_id=%s tool_name=%s is_error=%s",
            correlation_id,
            message_id,
            task_id,
            request.tool_id,
            request.tool_name,
            response.is_error,
        )

    logger.debug(
        "Notifying tool completion correlation_id=%s "
        "message_id=%s task_id=%s tool_id=%s tool_name=%s is_error=%s",
        correlation_id,
        message_id,
        task_id,
        request.tool_id,
        request.tool_name,
        response.is_error,
    )
    try:
        await _notify_completion(request, response)
    except Exception:
        logger.exception(
            "Tool completion notification failed correlation_id=%s message_id=%s "
            "tool_id=%s",
            correlation_id,
            message_id,
            request.tool_id,
        )
        raise
    finish_log = logger.error if response.is_error else logger.info
    finish_log(
        "Tool execution finished correlation_id=%s message_id=%s tool_id=%s "
        "is_error=%s result=%s",
        correlation_id,
        message_id,
        request.tool_id,
        response.is_error,
        _result_fragment(response),
    )
    return response.model_dump(mode="json")


async def _notify_completion(
    request: ToolExecutionRequest,
    response: ToolExecutionResponse,
) -> None:
    correlation_id = request.context.get("correlation_id")
    if not correlation_id or not request.tool_id:
        logger.debug(
            "Skipping tool completion correlation_id=%s "
            "message_id=%s tool_id=%s tool_name=%s",
            correlation_id,
            request.context.get("message_id") or correlation_id,
            request.tool_id,
            request.tool_name,
        )
        return
    scheduler = get_global_injector(True).get(ToolSchedulerFactory).get()
    await scheduler.complete(request, response)


def _result_fragment(response: ToolExecutionResponse) -> str:
    serialized = json.dumps(
        response.model_dump(mode="json")["result_content"],
        ensure_ascii=False,
        default=str,
    )
    single_line = " ".join(serialized.split())
    if len(single_line) <= RESULT_FRAGMENT_LENGTH:
        return single_line
    return f"{single_line[:RESULT_FRAGMENT_LENGTH]}..."


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
