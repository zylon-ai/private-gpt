"""BEFORE_TOOL/AFTER_TOOL interceptors and the tool action are one atomic unit.

Whatever fails inside that unit becomes an error ``tool_result``; it is never
propagated to the caller. The local scheduler and the Celery worker must
produce the same response for the same failure.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.components.tools.remote_execution import (
    ToolExecutionInterceptor,
    ToolExecutionRequest,
    ToolExecutionResponse,
    execute_tool_request,
)
from private_gpt.components.tools.tool_execution_outcome import ToolExecutionFailure
from private_gpt.components.tools.tool_scheduler import LocalToolScheduler

if TYPE_CHECKING:
    from private_gpt.components.tools.remote_execution import (
        ToolExecutionInterceptorContext,
    )


class _Exploding(ToolExecutionInterceptor):
    def __init__(self, phase: InterceptorPhase) -> None:
        self._phase = phase

    async def intercept(self, context: ToolExecutionInterceptorContext) -> None:
        if context.phase == self._phase:
            raise RuntimeError(f"boom in {self._phase.value}")


def _request(async_fn: Any = None) -> ToolExecutionRequest:
    async def echo(command: str) -> str:
        return f"ran {command}"

    spec = ToolSpec.from_defaults(
        name="bash",
        type="bash_v1",
        runtime="server",
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        async_fn=async_fn or echo,
    )
    return ToolExecutionRequest(
        tool_id="tool-1",
        tool_name="bash",
        tool_kwargs={"command": "ls"},
        tool_spec=spec,
        context={"correlation_id": "msg-1"},
    )


def _assert_error_response(
    response: ToolExecutionResponse, message: str, *, exception_type: str
) -> None:
    assert isinstance(response.outcome, ToolExecutionFailure)
    assert response.tool_id == "tool-1"
    assert response.tool_name == "bash"
    assert response.outcome.error.message == message
    assert response.outcome.error.exception_type == exception_type
    assert response.tool_message is not None
    assert response.tool_message.content == message
    assert response.tool_message.additional_kwargs["tool_call_id"] == "tool-1"
    assert response.tool_message.additional_kwargs["tool_call_name"] == "bash"
    assert response.tool_message.additional_kwargs["raw_output"] == message


@pytest.mark.anyio
@pytest.mark.parametrize(
    "phase", [InterceptorPhase.BEFORE_TOOL, InterceptorPhase.AFTER_TOOL]
)
async def test_interceptor_failure_becomes_error_tool_result(
    phase: InterceptorPhase,
) -> None:
    response = await execute_tool_request(_request(), interceptors=[_Exploding(phase)])

    _assert_error_response(
        response, f"boom in {phase.value}", exception_type="RuntimeError"
    )


@pytest.mark.anyio
async def test_tool_failure_becomes_error_tool_result() -> None:
    async def failing(command: str) -> str:
        raise ValueError(f"cannot run {command}")

    response = await execute_tool_request(_request(failing))

    assert isinstance(response.outcome, ToolExecutionFailure)
    assert "cannot run ls" in response.outcome.error.message
    assert response.tool_message is not None
    assert "cannot run ls" in str(response.tool_message.content)


@pytest.mark.anyio
async def test_local_and_celery_schedulers_return_the_same_error_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_module = importlib.import_module(
        "private_gpt.celery.tasks.tools.tool_run_task"
    )
    scheduler_module = importlib.import_module(
        "private_gpt.components.tools.tool_scheduler"
    )

    request = _request()
    for module in (task_module, scheduler_module):
        monkeypatch.setattr(
            module,
            "execute_tool_request",
            AsyncMock(side_effect=RuntimeError("boom: executor crashed")),
        )
    monkeypatch.setattr(
        task_module, "_claim_tool_execution", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(task_module, "_notify_completion", AsyncMock())

    local = await LocalToolScheduler().execute(request)
    remote = ToolExecutionResponse.model_validate(
        await task_module.tool_run_task.run(
            request_data=request.model_dump(mode="json")
        )
    )

    _assert_error_response(
        local, "boom: executor crashed", exception_type="RuntimeError"
    )
    assert local.model_dump(mode="json") == remote.model_dump(mode="json")


@pytest.mark.anyio
async def test_local_scheduler_never_raises_when_execution_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "private_gpt.components.tools.tool_scheduler.execute_tool_request",
        AsyncMock(side_effect=RuntimeError("boom: executor crashed")),
    )

    response = await LocalToolScheduler().execute(_request())

    _assert_error_response(
        response, "boom: executor crashed", exception_type="RuntimeError"
    )
