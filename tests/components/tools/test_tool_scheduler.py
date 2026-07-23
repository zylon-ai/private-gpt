import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.tools.remote_execution import ToolExecutionRequest
from private_gpt.components.tools.tool_scheduler import (
    TOOL_TASK_NAME,
    CeleryToolScheduler,
    LocalToolScheduler,
)
from private_gpt.events.models import TextBlock


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

    with (
        caplog.at_level(
            logging.ERROR,
            logger="private_gpt.components.tools.tool_scheduler",
        ),
        pytest.raises(RuntimeError, match="boom"),
    ):
        await LocalToolScheduler().execute(tool_request())

    assert "Local tool 'bash' execution failed" in caplog.text
    assert "RuntimeError: boom" in caplog.text


@pytest.mark.anyio
async def test_celery_tool_scheduler_execute_dispatches_and_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async_result = MagicMock()
    dispatch_task = MagicMock(return_value=async_result)
    response_data = {
        "tool_name": "bash",
        "tool_id": "tool-1",
        "result_content": [TextBlock(text="worker result").model_dump(mode="json")],
        "is_error": False,
        "tool_message": {
            "role": "tool",
            "content": "worker result",
            "additional_kwargs": {"tool_call_id": "tool-1"},
        },
    }
    to_thread = AsyncMock(return_value=response_data)
    monkeypatch.setattr(
        "private_gpt.components.tools.tool_scheduler.dispatch_task", dispatch_task
    )
    monkeypatch.setattr(
        "private_gpt.components.tools.tool_scheduler.to_thread", to_thread
    )
    scheduler = CeleryToolScheduler(
        settings=SimpleNamespace(
            scheduler=SimpleNamespace(
                tools=SimpleNamespace(
                    celery_queue="tools",
                    callback_timeout_seconds=42,
                )
            )
        ),
    )
    request = tool_request()

    response = await scheduler.execute(request)

    assert response.result_content == [TextBlock(text="worker result")]
    dispatch_task.assert_called_once_with(
        task_name=TOOL_TASK_NAME,
        kwargs={"request_data": request.model_dump(mode="json")},
        queue="tools",
        ignore_result=False,
    )
    to_thread.assert_awaited_once_with(async_result.get, timeout=42)


@pytest.mark.anyio
async def test_celery_tool_scheduler_async_execute_ignores_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async_result = MagicMock(id="task-abc")
    dispatch_task = MagicMock(return_value=async_result)
    monkeypatch.setattr(
        "private_gpt.components.tools.tool_scheduler.dispatch_task", dispatch_task
    )
    scheduler = CeleryToolScheduler(
        settings=SimpleNamespace(
            scheduler=SimpleNamespace(
                tools=SimpleNamespace(celery_queue="tools"),
            )
        ),
    )
    request = tool_request()

    task_id = await scheduler.async_execute(request)

    assert task_id == "task-abc"
    dispatch_task.assert_called_once_with(
        task_name=TOOL_TASK_NAME,
        kwargs={"request_data": request.model_dump(mode="json")},
        queue="tools",
        task_id="msg-1:tool-1",
        ignore_result=True,
    )


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
    async with asyncio.timeout(2):
        while not celery_app.control.revoke.called:
            await asyncio.sleep(0.01)
    celery_app.control.revoke.assert_called_once_with("task-abc", terminate=True)
