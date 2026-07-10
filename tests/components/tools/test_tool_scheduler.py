import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.tools.remote_execution import ToolExecutionRequest
from private_gpt.components.tools.tool_scheduler import CeleryToolScheduler


class _Result:
    def __init__(self, task_id: str) -> None:
        self.id = task_id
        self.result = None

    def ready(self) -> bool:
        return False

    def failed(self) -> bool:
        return False


@pytest.mark.anyio
async def test_celery_tool_scheduler_cancels_child_task_when_stream_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch_task = MagicMock(return_value=_Result("msg-1:tool-1"))
    monkeypatch.setattr(
        "private_gpt.components.tools.tool_scheduler.dispatch_task",
        dispatch_task,
    )

    celery_app = MagicMock()
    monkeypatch.setattr("private_gpt.celery.celery.celery_app", celery_app)

    stream_component = SimpleNamespace(
        stream=SimpleNamespace(is_cancelled=AsyncMock(return_value=True))
    )
    scheduler = CeleryToolScheduler(
        settings=SimpleNamespace(
            scheduler=SimpleNamespace(tools=SimpleNamespace(celery_queue="tools"))
        ),
        stream_component=stream_component,
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

    with pytest.raises(asyncio.CancelledError):
        await scheduler.execute(request)

    celery_app.control.revoke.assert_called_once_with("msg-1:tool-1", terminate=True)
