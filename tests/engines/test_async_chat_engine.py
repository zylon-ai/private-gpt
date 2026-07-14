import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import ToolSelection

from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
    ResolvedSystemConfig,
    ResolvedToolConfig,
    ToolSpec,
)
from private_gpt.components.engines.chat.async_chat_engine import (
    AsyncChatEngine,
    IterationCheckpointPayload,
    LocalEventChannel,
)
from private_gpt.components.engines.chat.chat_engine import ChatLoopEngine
from private_gpt.components.engines.chat.models.chat_state import (
    ChatState,
    ChatStatus,
)
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.tools.remote_execution import (
    ToolExecutionRequest,
    ToolExecutionResponse,
    build_rebuild_metadata,
    execute_tool_request,
)
from private_gpt.components.tools.tool_scheduler import (
    BaseToolScheduler,
    LocalToolScheduler,
)
from private_gpt.events.models import (
    RawContentBlockStartEvent,
    TextBlock,
    ToolResultBlock,
)
from tests.fixtures.mock_function_llm import get_mock_function_calling_llm


async def _noop_tool(value: str) -> str:
    await asyncio.sleep(0.01)
    return f"ok:{value}"


def _client_tool(name: str) -> ToolSpec:
    return ToolSpec.from_defaults(
        name=name,
        type=name,
        runtime="client",
        input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
    )


def _rebuild_server_tool(name: str) -> ToolSpec:
    return ToolSpec.from_defaults(
        name=name,
        type=name,
        runtime="server",
        async_fn=_noop_tool,
    )


def _server_tool(name: str) -> ToolSpec:
    return ToolSpec.from_defaults(
        name=name,
        type=name,
        runtime="server",
        async_fn=_noop_tool,
        execution_metadata=build_rebuild_metadata(
            _rebuild_server_tool,
            {"name": name},
        ),
    )


@pytest.fixture
def base_request() -> ResolvedChatRequest:
    return ResolvedChatRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="hello")],
        system=ResolvedSystemConfig(prompt="test"),
    )


class _FakeAsyncToolScheduler(BaseToolScheduler):
    def __init__(self) -> None:
        self.pending: dict[str, tuple[ToolExecutionRequest, str]] = {}
        self.cancelled: list[str] = []
        self._next = 0

    @property
    def is_async(self) -> bool:
        return True

    async def execute(
        self,
        request: ToolExecutionRequest,
        state_ctx=None,
        interceptors=None,
    ) -> ToolExecutionResponse:
        del request, state_ctx, interceptors
        raise NotImplementedError

    async def async_execute(
        self,
        request: ToolExecutionRequest,
        state_ctx=None,
        interceptors=None,
    ) -> str:
        del state_ctx, interceptors
        self._next += 1
        handle = f"handle-{self._next}"
        self.pending[request.tool_id] = (request, handle)
        return handle

    async def cancel(
        self,
        request: ToolExecutionRequest,
        task_id: str | None = None,
    ) -> bool:
        del request
        if task_id is None:
            return False
        self.cancelled.append(task_id)
        return True

    async def complete_pending(self) -> list[ToolExecutionResponse]:
        responses: list[ToolExecutionResponse] = []
        for tool_id in list(self.pending):
            request, _ = self.pending.pop(tool_id)
            responses.append(await execute_tool_request(request))
        return responses


class _FakeChatScheduler:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    async def cancel(self, correlation_id: str) -> bool:
        self.cancelled.append(correlation_id)
        return True


@dataclass
class _AsyncRunResult:
    events: list[Any]
    states: list[ChatState]


def _make_llm_component(mock_llm: Any) -> MagicMock:
    llm_component = MagicMock(spec=LLMComponent)
    llm_component.get_llm.return_value = mock_llm
    return llm_component


async def _collect_events(events: AsyncGenerator[Any, None]) -> list[Any]:
    return [event async for event in events]


def _normalize(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {
            key: _normalize(item)
            for key, item in value.items()
            if key
            not in {
                "block_id",
                "id",
                "start_timestamp",
                "stop_timestamp",
                "expires_at",
                "tool_use_id",
            }
        }
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _normalize_events(events: list[Any]) -> list[Any]:
    return [
        {
            "type": event.__class__.__name__,
            "payload": _normalize(event),
        }
        for event in events
    ]


def _tool_result_texts(events: list[Any]) -> list[str]:
    texts: list[str] = []
    for event in events:
        if isinstance(event, RawContentBlockStartEvent) and isinstance(
            event.content_block, ToolResultBlock
        ):
            for block in event.content_block.content:
                if isinstance(block, TextBlock):
                    texts.append(block.text)
    return texts


async def _run_sync_engine(
    request: ResolvedChatRequest,
    mock_llm: Any,
    tool_scheduler: BaseToolScheduler | None = None,
) -> list[Any]:
    engine = ChatLoopEngine(
        llm_component=_make_llm_component(mock_llm),
        request_interceptors=[],
        response_interceptors=[],
        max_iterations=6,
        tool_scheduler=tool_scheduler or LocalToolScheduler(),
    )
    execution = await engine.run(request)
    events = await _collect_events(execution.events)
    await execution.final_state_task
    return events


async def _drain(channel: LocalEventChannel) -> list[Any]:
    """Drain all events from a closed LocalEventChannel."""
    return [e async for e in channel.stream()]


async def _run_async_engine(
    request: ResolvedChatRequest,
    mock_llm: Any,
    tool_scheduler: BaseToolScheduler,
) -> _AsyncRunResult:
    engine = AsyncChatEngine(
        llm_component=_make_llm_component(mock_llm),
        request_interceptors=[],
        response_interceptors=[],
        max_iterations=6,
        tool_scheduler=tool_scheduler,
        chat_scheduler=_FakeChatScheduler(),
    )

    all_events: list[Any] = []
    states: list[ChatState] = []

    channel = LocalEventChannel()
    state = await engine.execute(request, channel=channel)
    await channel.close()
    all_events.extend(await _drain(channel))
    states.append(state)

    while state.output.status == ChatStatus.WAITING:
        assert isinstance(tool_scheduler, _FakeAsyncToolScheduler)
        responses = await tool_scheduler.complete_pending()
        resumed_request = state.input.request.model_copy(deep=True)
        resumed_request.messages = [
            *resumed_request.messages,
            *(response.tool_message for response in responses),
        ]
        channel2 = LocalEventChannel()
        state = await engine.resume(
            state.output.pause_type,
            resumed_request,
            iteration=state.runtime.iteration,
            next_block_count=state.runtime.next_block_count,
            checkpoint_payload=IterationCheckpointPayload(
                pending_async_tools=state.output.pending_async_tools,
                tool_responses=responses,
                pending_external_tool_calls=state.output.pending_external_tool_calls,
                total_input_tokens=state.runtime.total_input_tokens,
                total_output_tokens=state.runtime.total_output_tokens,
                has_input_usage=state.runtime.has_input_usage,
                has_output_usage=state.runtime.has_output_usage,
            ),
            channel=channel2,
        )
        await channel2.close()
        all_events.extend(await _drain(channel2))
        states.append(state)

    return _AsyncRunResult(events=all_events, states=states)


@pytest.mark.asyncio
async def test_async_engine_matches_sync_simple_message(
    base_request: ResolvedChatRequest,
) -> None:
    sync_events = await _run_sync_engine(
        base_request.model_copy(deep=True),
        get_mock_function_calling_llm(["hello", " world"]),
    )
    async_result = await _run_async_engine(
        base_request.model_copy(deep=True),
        get_mock_function_calling_llm(["hello", " world"]),
        tool_scheduler=_FakeAsyncToolScheduler(),
    )

    assert _normalize_events(async_result.events) == _normalize_events(sync_events)
    assert [state.output.status for state in async_result.states] == [
        ChatStatus.COMPLETED,
    ]


@pytest.mark.asyncio
async def test_async_engine_matches_sync_one_client_tool_and_stops_first_iteration(
    base_request: ResolvedChatRequest,
) -> None:
    request = base_request.model_copy(deep=True)
    request.tool_config = ResolvedToolConfig(tools=[_client_tool("browser")])
    sync_events = await _run_sync_engine(
        request.model_copy(deep=True),
        get_mock_function_calling_llm(
            [
                [
                    ToolSelection(
                        tool_id="tool_1",
                        tool_name="browser",
                        tool_kwargs={"value": "x"},
                    )
                ]
            ]
        ),
    )
    async_result = await _run_async_engine(
        request.model_copy(deep=True),
        get_mock_function_calling_llm(
            [
                [
                    ToolSelection(
                        tool_id="tool_1",
                        tool_name="browser",
                        tool_kwargs={"value": "x"},
                    )
                ]
            ]
        ),
        tool_scheduler=_FakeAsyncToolScheduler(),
    )

    assert _normalize_events(async_result.events) == _normalize_events(sync_events)
    assert async_result.states[0].output.status == ChatStatus.COMPLETED
    assert async_result.states[0].output.stop_reason == "tool_use"
    assert len(async_result.states[0].output.pending_external_tool_calls) == 1


@pytest.mark.asyncio
async def test_async_engine_matches_sync_one_server_tool_and_resumes_same_point(
    base_request: ResolvedChatRequest,
) -> None:
    request = base_request.model_copy(deep=True)
    request.tool_config = ResolvedToolConfig(tools=[_server_tool("echo")])
    sync_events = await _run_sync_engine(
        request.model_copy(deep=True),
        get_mock_function_calling_llm(
            [
                [
                    ToolSelection(
                        tool_id="tool_1", tool_name="echo", tool_kwargs={"value": "x"}
                    )
                ],
                ["done"],
            ]
        ),
    )
    async_result = await _run_async_engine(
        request.model_copy(deep=True),
        get_mock_function_calling_llm(
            [
                [
                    ToolSelection(
                        tool_id="tool_1", tool_name="echo", tool_kwargs={"value": "x"}
                    )
                ],
                ["done"],
            ]
        ),
        tool_scheduler=_FakeAsyncToolScheduler(),
    )

    assert [state.output.status for state in async_result.states] == [
        ChatStatus.WAITING,
        ChatStatus.COMPLETED,
    ]
    assert async_result.states[0].output.pause_type == "tools"
    assert _tool_result_texts(async_result.events) == ["ok:x"]
    assert _tool_result_texts(sync_events) == ["ok:x"]
    assert _normalize_events(async_result.events) == _normalize_events(sync_events)


@pytest.mark.asyncio
async def test_async_engine_matches_sync_two_server_tools_plus_one_client_tool(
    base_request: ResolvedChatRequest,
) -> None:
    request = base_request.model_copy(deep=True)
    request.tool_config = ResolvedToolConfig(
        tools=[
            _server_tool("echo"),
            _server_tool("echo2"),
            _client_tool("browser"),
        ]
    )
    sync_events = await _run_sync_engine(
        request.model_copy(deep=True),
        get_mock_function_calling_llm(
            [
                [
                    ToolSelection(
                        tool_id="tool_1", tool_name="echo", tool_kwargs={"value": "a"}
                    ),
                    ToolSelection(
                        tool_id="tool_2", tool_name="echo2", tool_kwargs={"value": "b"}
                    ),
                    ToolSelection(
                        tool_id="tool_3",
                        tool_name="browser",
                        tool_kwargs={"value": "c"},
                    ),
                ]
            ]
        ),
    )
    async_result = await _run_async_engine(
        request.model_copy(deep=True),
        get_mock_function_calling_llm(
            [
                [
                    ToolSelection(
                        tool_id="tool_1", tool_name="echo", tool_kwargs={"value": "a"}
                    ),
                    ToolSelection(
                        tool_id="tool_2", tool_name="echo2", tool_kwargs={"value": "b"}
                    ),
                    ToolSelection(
                        tool_id="tool_3",
                        tool_name="browser",
                        tool_kwargs={"value": "c"},
                    ),
                ]
            ]
        ),
        tool_scheduler=_FakeAsyncToolScheduler(),
    )

    assert async_result.states[0].output.status == ChatStatus.WAITING
    assert async_result.states[1].output.status == ChatStatus.COMPLETED
    assert async_result.states[1].output.stop_reason == "tool_use"
    assert len(async_result.states[1].output.pending_external_tool_calls) == 1
    assert sorted(_tool_result_texts(async_result.events)) == ["ok:a", "ok:b"]
    assert sorted(_tool_result_texts(sync_events)) == ["ok:a", "ok:b"]
    assert _normalize_events(async_result.events) == _normalize_events(sync_events)


@pytest.mark.asyncio
async def test_async_engine_matches_sync_across_multiple_server_tool_iterations(
    base_request: ResolvedChatRequest,
) -> None:
    request = base_request.model_copy(deep=True)
    request.tool_config = ResolvedToolConfig(tools=[_server_tool("echo")])
    deltas = [
        [ToolSelection(tool_id="tool_1", tool_name="echo", tool_kwargs={"value": "a"})],
        [ToolSelection(tool_id="tool_2", tool_name="echo", tool_kwargs={"value": "b"})],
        ["all", " done"],
    ]
    sync_events = await _run_sync_engine(
        request.model_copy(deep=True),
        get_mock_function_calling_llm(deltas),
    )
    async_result = await _run_async_engine(
        request.model_copy(deep=True),
        get_mock_function_calling_llm(deltas),
        tool_scheduler=_FakeAsyncToolScheduler(),
    )

    assert [state.output.status for state in async_result.states] == [
        ChatStatus.WAITING,
        ChatStatus.WAITING,
        ChatStatus.COMPLETED,
    ]
    assert _tool_result_texts(async_result.events) == ["ok:a", "ok:b"]
    assert _normalize_events(async_result.events) == _normalize_events(sync_events)


@pytest.mark.asyncio
async def test_async_engine_cancel_schedules_chat_cancellation(
    base_request: ResolvedChatRequest,
) -> None:
    request = base_request.model_copy(deep=True)
    request.context.correlation_id = "msg-cancel-1"
    request.tool_config = ResolvedToolConfig(tools=[_server_tool("echo")])
    mock_llm = get_mock_function_calling_llm(
        [
            [
                ToolSelection(
                    tool_id="tool_1", tool_name="echo", tool_kwargs={"value": "x"}
                )
            ]
        ]
    )
    tool_scheduler = _FakeAsyncToolScheduler()
    chat_scheduler = _FakeChatScheduler()
    engine = AsyncChatEngine(
        llm_component=_make_llm_component(mock_llm),
        request_interceptors=[],
        response_interceptors=[],
        max_iterations=4,
        tool_scheduler=tool_scheduler,
        chat_scheduler=chat_scheduler,
    )

    channel = LocalEventChannel()
    state = await engine.execute(request, channel=channel)
    await channel.close()
    await _collect_events(channel.stream())

    assert state.output.status == ChatStatus.WAITING
    assert list(state.output.pending_async_tools.values()) == ["handle-1"]

    cancelled = await engine.cancel("msg-cancel-1")

    assert cancelled is True
    assert chat_scheduler.cancelled == ["msg-cancel-1"]


@pytest.mark.asyncio
async def test_async_engine_cancel_without_scheduler_returns_false() -> None:
    engine = AsyncChatEngine(
        llm_component=_make_llm_component(get_mock_function_calling_llm(["ok"])),
        request_interceptors=[],
        response_interceptors=[],
        max_iterations=2,
        tool_scheduler=_FakeAsyncToolScheduler(),
        chat_scheduler=_FakeChatScheduler(),
    )

    # _FakeChatScheduler returns True for any correlation_id
    assert await engine.cancel("msg-no-scheduler") is True
