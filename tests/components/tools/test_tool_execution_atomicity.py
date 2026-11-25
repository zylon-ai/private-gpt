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
from private_gpt.components.tools.builders import web_search_builder
from private_gpt.components.tools.remote_execution import (
    ToolExecutionInterceptor,
    ToolExecutionRequest,
    ToolExecutionResponse,
    build_rebuild_metadata,
    execute_tool_request,
)
from private_gpt.components.tools.tool_execution_outcome import (
    ToolExecutionFailure,
    ToolExecutionSuccess,
)
from private_gpt.components.tools.tool_scheduler import LocalToolScheduler
from private_gpt.components.web.web_search.models import WebSearchResult
from private_gpt.components.web.web_search.providers.brave import BraveSearchProvider
from private_gpt.components.web.web_search.providers.parallel import (
    ParallelSearchProvider,
)
from tests.fixtures.mock_injector import MockInjector

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
@pytest.mark.parametrize(
    ("tool_type", "api_provider", "worker_provider", "expected_error"),
    [
        (
            "parallel_web_search_v1",
            "parallel",
            "brave",
            "differs from the expected provider",
        ),
        ("web_search_v0", "parallel", "brave", "Review the current provider"),
        ("web_search_v1", "parallel", "auto", "Select the search provider"),
        (
            "explicit_parallel_web_search_v1",
            "parallel",
            "auto",
            "provider selection changed",
        ),
        ("explicit_parallel_web_search_v1", "parallel", "parallel", None),
        (
            "automatic_parallel_web_search_v1",
            "auto",
            "parallel",
            "provider selection changed",
        ),
        ("automatic_parallel_web_search_v1", "auto", "auto", None),
    ],
)
async def test_worker_honors_the_search_provider_in_a_queued_tool(
    injector: MockInjector,
    monkeypatch: pytest.MonkeyPatch,
    tool_type: str,
    api_provider: str,
    worker_provider: str,
    expected_error: str | None,
) -> None:
    task_module = importlib.import_module(
        "private_gpt.celery.tasks.tools.tool_run_task"
    )
    injector.bind_settings(
        {
            "web_search": {
                "enabled": True,
                "provider": api_provider,
                "processor": "simple_text",
            }
        }
    )
    spec = await injector.get(web_search_builder.WebSearchToolBuilder).build_tool(
        type=tool_type
    )
    if tool_type == "web_search_v0":
        assert spec.execution_metadata is not None
        # An older API recognizes v0 as web search and queues the generic rebuild.
        spec.execution_metadata = build_rebuild_metadata(
            web_search_builder.rebuild_web_search_tool,
            spec.execution_metadata.rebuild_kwargs,
        )
    request = ToolExecutionRequest(
        tool_id="search-1",
        tool_name="web_search",
        tool_kwargs={"query": "Cassini spacecraft"},
        tool_spec=spec,
    )
    worker_injector = MockInjector()
    worker_injector.bind_settings(
        {
            "web_search": {
                "enabled": True,
                "provider": worker_provider,
                "processor": "simple_text",
            },
            "brave": {"api_key": "fixture-key"},
        }
    )
    monkeypatch.setattr(
        web_search_builder, "get_global_injector", lambda: worker_injector.test_injector
    )
    search = AsyncMock(
        return_value=[
            WebSearchResult(
                url="https://example.org/cassini",
                title="Cassini",
                description="Cassini excerpt",
            )
        ]
    )
    monkeypatch.setattr(BraveSearchProvider, "make_query", search)
    monkeypatch.setattr(ParallelSearchProvider, "make_query", search)

    result = ToolExecutionResponse.model_validate(
        await task_module.tool_run_task.run(
            request_data=request.model_dump(mode="json")
        )
    )

    if expected_error is None:
        search.assert_awaited_once()
        assert search.await_args.args[0] == "Cassini spacecraft"
        assert isinstance(result.outcome, ToolExecutionSuccess)
    else:
        search.assert_not_awaited()
        assert isinstance(result.outcome, ToolExecutionFailure)
        assert expected_error in result.outcome.error.message


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
