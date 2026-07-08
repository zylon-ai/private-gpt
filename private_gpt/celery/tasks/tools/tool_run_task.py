"""Celery task that executes a single tool call on a dedicated tools worker.

When ``tool_scheduler.mode`` is ``"celery"``, the chat worker dispatches tool
calls to this task on the ``tools`` queue instead of running them in-process.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from private_gpt.celery.base import StatefulBackgroundTask
from private_gpt.celery.celery import celery_app
from private_gpt.di import get_global_injector

if TYPE_CHECKING:
    from private_gpt.server.tools.tool_service import ToolService

logger = logging.getLogger(__name__)


@celery_app.task(
    name="private_gpt.tools.run",
    base=StatefulBackgroundTask,
)
async def tool_run_task(
    tool_name: str,
    chat_priority: int | None,
    tool_id: str,
    tool_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Execute a tool call and return a serializable result.

    Receives the tool name and raw keyword arguments, looks up the tool
    from the warm dependency injector, and executes it. Returns a dict
    with ``content`` and ``is_error`` keys.

    Args:
        tool_name: The registered tool name (e.g. ``"semantic_search"``).
        chat_priority: Priority signal from the chat request (unused here
            but available for future scheduling).
        tool_id: The tool call ID assigned by the LLM.
        tool_kwargs: Keyword arguments for the tool function.
    """
    injector = get_global_injector()

    from private_gpt.server.tools.tool_service import ToolService

    tool_service = injector.get(ToolService)

    try:
        result = await _execute_by_name(tool_service, tool_name, tool_kwargs)
        return {
            "tool_name": tool_name,
            "tool_id": tool_id,
            "content": result.content,
            "is_error": result.is_error,
        }
    except Exception as e:
        logger.exception("Tool '%s' execution failed", tool_name)
        return {
            "tool_name": tool_name,
            "tool_id": tool_id,
            "content": [{"type": "text", "text": str(e)}],
            "is_error": True,
        }


async def _execute_by_name(
    tool_service: ToolService,
    tool_name: str,
    kwargs: dict[str, Any],
) -> Any:
    """Dispatch to the appropriate ToolService method by tool name."""
    from private_gpt.chat.extensions.context_filter import ContextFilter

    tool_map: dict[str, str] = {
        "semantic_search": "semantic_search_tool",
        "web_search": "web_search_tool",
        "web_fetch": "web_fetch_tool",
        "database_query": "database_query_tool",
        "tabular_data_analysis": "tabular_data_analysis_tool",
    }

    method_name = tool_map.get(tool_name)
    if method_name is None:
        raise ValueError(f"Unsupported tool for remote execution: {tool_name!r}")

    method = getattr(tool_service, method_name)

    if "context_filter" not in kwargs and tool_name in (
        "semantic_search",
        "tabular_data_analysis",
    ):
        kwargs["context_filter"] = ContextFilter()

    return await method(**kwargs)
