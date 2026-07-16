import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.tools.remote_execution import ToolExecutionRequest
from private_gpt.components.tools.tool_scheduler import (
    CeleryToolScheduler,
    LocalToolScheduler,
)


def tool_request() -> ToolExecutionRequest:
    return ToolExecutionRequest.model_validate(
        {
            "tool_id": "tool-1",
            "tool_name": "bash",
            "tool_kwargs": {},
            "tool_spec": ToolSpec(name="bash", runtime="server", input_schema={}),
            "context": {"correlation_id": "msg-1", "messages": []},
        }
    )


@pytest.mark.anyio
async def test_local_tool_scheduler_logs_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    execute_tool_request = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(
        "private_gpt.components.tools.tool_scheduler.execute_tool_request",
        execute_tool_request,
    )

    with caplog.at_level(
        logging.ERROR,
        logger="private_gpt.components.tools.tool_scheduler",
    ), pytest.raises(RuntimeError, match="boom"):
        await LocalToolScheduler().execute(tool_request())

    assert "Local tool 'bash' execution failed" in caplog.text
    assert "RuntimeError: boom" in caplog.text


@pytest.mark.anyio
async def test_celery_tool_scheduler_execute_raises_not_implemented() -> None:
    scheduler = CeleryToolScheduler(
        settings=SimpleNamespace(
            scheduler=SimpleNamespace(tools=SimpleNamespace(celery_queue="tools"))
        ),
    )

    with pytest.raises(NotImplementedError):
        await scheduler.execute(tool_request())


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

    cancelled = await scheduler.cancel(tool_request(), task_id="task-abc")

    assert cancelled is True
    celery_app.control.revoke.assert_called_once_with("task-abc", terminate=True)
