import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.celery.tasks.tools.tool_run_task import (
    _claim_tool_execution,
    _notify_completion,
)
from private_gpt.components.engines.chat.models.execution_hooks import ExecutionHooks
from private_gpt.components.tools.remote_execution import (
    ToolExecutionRequest,
    ToolExecutionResponse,
)
from private_gpt.events.models import TextBlock


@pytest.mark.anyio
async def test_duplicate_tool_execution_is_claimed_once_across_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MagicMock()
    store.claim_action = AsyncMock(side_effect=[True, False])
    store_factory = MagicMock()
    store_factory.get.return_value = store
    injector = MagicMock()
    injector.get.return_value = store_factory
    task_module = importlib.import_module(
        "private_gpt.celery.tasks.tools.tool_run_task"
    )
    monkeypatch.setattr(
        task_module, "get_global_injector", MagicMock(return_value=injector)
    )
    request = ToolExecutionRequest.model_validate(
        {
            "tool_id": "tool-1",
            "tool_name": "charge_customer",
            "tool_kwargs": {},
            "tool_spec": {
                "name": "charge_customer",
                "runtime": "server",
                "input_schema": {},
            },
            "context": {"correlation_id": "chat-1"},
        }
    )

    assert await _claim_tool_execution(request) is True
    assert await _claim_tool_execution(request) is False
    assert store.claim_action.await_count == 2
    store.claim_action.assert_awaited_with("chat-1", "tool:tool-1")


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
