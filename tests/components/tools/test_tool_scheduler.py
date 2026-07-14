from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.tools.remote_execution import ToolExecutionRequest
from private_gpt.components.tools.tool_scheduler import CeleryToolScheduler


@pytest.mark.anyio
async def test_celery_tool_scheduler_execute_raises_not_implemented() -> None:
    scheduler = CeleryToolScheduler(
        settings=SimpleNamespace(
            scheduler=SimpleNamespace(tools=SimpleNamespace(celery_queue="tools"))
        ),
    )

    request = ToolExecutionRequest.model_validate(
        {
            "tool_id": "tool-1",
            "tool_name": "bash",
            "tool_kwargs": {},
            "tool_spec": ToolSpec(name="bash", runtime="server", input_schema={}),
            "context": {"correlation_id": "msg-1", "messages": []},
        }
    )

    with pytest.raises(NotImplementedError):
        await scheduler.execute(request)


@pytest.mark.anyio
async def test_celery_tool_scheduler_cancel_revokes_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    celery_app = MagicMock()
    monkeypatch.setattr("private_gpt.celery.celery.celery_app", celery_app)

    scheduler = CeleryToolScheduler(
        settings=SimpleNamespace(
            scheduler=SimpleNamespace(tools=SimpleNamespace(celery_queue="tools"))
        ),
    )

    request = ToolExecutionRequest.model_validate(
        {
            "tool_id": "tool-1",
            "tool_name": "bash",
            "tool_kwargs": {},
            "tool_spec": ToolSpec(name="bash", runtime="server", input_schema={}),
            "context": {"correlation_id": "msg-1", "messages": []},
        }
    )

    cancelled = await scheduler.cancel(request, task_id="task-abc")

    assert cancelled is True
    celery_app.control.revoke.assert_called_once_with("task-abc", terminate=True)
