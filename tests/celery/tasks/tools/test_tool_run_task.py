import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.celery.tasks.tools.tool_run_task import _notify_completion
from private_gpt.components.engines.chat.models.execution_hooks import ExecutionHooks
from private_gpt.components.tools.remote_execution import (
    ToolExecutionRequest,
    ToolExecutionResponse,
)
from private_gpt.events.models import TextBlock


@pytest.mark.anyio
async def test_notify_completion_propagates_callback_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = MagicMock()
    scheduler.complete = AsyncMock(side_effect=RuntimeError("ARQ enqueue failed"))
    scheduler_factory = MagicMock()
    scheduler_factory.get.return_value = scheduler
    injector = MagicMock()
    injector.get.return_value = scheduler_factory
    task_module = importlib.import_module(
        "private_gpt.celery.tasks.tools.tool_run_task"
    )
    monkeypatch.setattr(
        task_module, "get_global_injector", MagicMock(return_value=injector)
    )
    request = ToolExecutionRequest.model_validate(
        {
            "tool_id": "semantic-search-1",
            "tool_name": "semantic_search",
            "tool_kwargs": {},
            "tool_spec": {
                "name": "semantic_search",
                "runtime": "server",
                "input_schema": {},
            },
            "context": {"correlation_id": "chat-1"},
            "hooks": ExecutionHooks(),
        }
    )
    response = ToolExecutionResponse(
        tool_name="semantic_search",
        tool_id="semantic-search-1",
        result_content=[TextBlock(text="query: Field required")],
        is_error=True,
        tool_message={
            "role": "tool",
            "content": "query: Field required",
            "additional_kwargs": {"tool_call_id": "semantic-search-1"},
        },
    )

    with pytest.raises(RuntimeError, match="ARQ enqueue failed"):
        await _notify_completion(request, response)
