import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.schema import NodeWithScore, TextNode
from pydantic import Field

from private_gpt.components.chat.models.chat_config_models import (
    CitationConfig,
    ResolvedChatRequest,
    ResolvedSystemConfig,
    ResolvedToolConfig,
    ToolSpec,
)
from private_gpt.components.engines.chat.async_chat_engine import (
    AsyncChatCheckpoint,
    AsyncChatEngine,
    IterationCheckpointPayload,
    LocalEventChannel,
)
from private_gpt.components.engines.chat.chat_engine import ChatLoopEngine
from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.components.engines.chat.models.chat_state import (
    ChatInputState,
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
from private_gpt.server.chat.interceptors.citation_interceptor import (
    CitationRequestInterceptor,
)
from private_gpt.server.chat.interceptors.document_processing_interceptor import (
    DocumentProcessingRequestInterceptor,
)
from private_gpt.server.chat.interceptors.extract_citation_interceptor import (
    ExtractCitationInterceptor,
)
from private_gpt.server.chat.interceptors.runtime_model_interceptor import (
    RuntimeModelRequestInterceptor,
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
class _RuntimeObservation:
    phase: InterceptorPhase
    model_id: str | None
    effective_token_limit: int | None
    has_tokenizer: bool


class _RuntimeRecordingInterceptor(ChatRequestLoopInterceptor):
    observations: list[_RuntimeObservation]

    async def intercept(self, context: ChatInterceptorContext) -> None:
        self.observations.append(
            _RuntimeObservation(
                phase=context.phase,
                model_id=context.state.runtime.model_id,
                effective_token_limit=context.state.runtime.effective_token_limit,
                has_tokenizer=context.state.runtime.tokenizer_fn is not None,
            )
        )


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
    request_interceptors: list[ChatRequestLoopInterceptor] | None = None,
    llm_component: LLMComponent | None = None,
) -> _AsyncRunResult:
    resolved_llm_component = llm_component or _make_llm_component(mock_llm)
    engine = AsyncChatEngine(
        llm_component=resolved_llm_component,
        request_interceptors=request_interceptors or [],
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
            AsyncChatCheckpoint(
                checkpoint=state.output.pause_type,
                input=ChatInputState(
                    request=resumed_request,
                    context_stack=state.input.context_stack,
                ),
                iteration=state.runtime.iteration,
                next_block_count=state.runtime.next_block_count,
                payload=IterationCheckpointPayload(
                    model_id=state.runtime.model_id,
                    pending_async_tools=state.output.pending_async_tools,
                    tool_responses=responses,
                    pending_external_tool_calls=state.output.pending_external_tool_calls,
                    total_input_tokens=state.runtime.total_input_tokens,
                    total_output_tokens=state.runtime.total_output_tokens,
                    has_input_usage=state.runtime.has_input_usage,
                    has_output_usage=state.runtime.has_output_usage,
                ),
            ),
            channel=channel2,
        )
        await channel2.close()
        all_events.extend(await _drain(channel2))
        states.append(state)

    return _AsyncRunResult(events=all_events, states=states)


class _RecordingRequestInterceptor(ChatRequestLoopInterceptor):
    observations: list[tuple[InterceptorPhase, list[MessageRole]]]

    async def intercept(self, context: ChatInterceptorContext) -> None:
        self.observations.append(
            (
                context.phase,
                [message.role for message in context.state.input.request.messages],
            )
        )


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
async def test_async_engine_reruns_before_iteration_with_resumed_tool_results(
    base_request: ResolvedChatRequest,
) -> None:
    request = base_request.model_copy(deep=True)
    request.tool_config = ResolvedToolConfig(tools=[_server_tool("echo")])
    recorder = _RecordingRequestInterceptor(observations=[])

    await _run_async_engine(
        request,
        get_mock_function_calling_llm(
            [
                [
                    ToolSelection(
                        tool_id="tool_1",
                        tool_name="echo",
                        tool_kwargs={"value": "x"},
                    )
                ],
                ["done"],
            ]
        ),
        tool_scheduler=_FakeAsyncToolScheduler(),
        request_interceptors=[recorder],
    )

    before_iteration_roles = [
        roles
        for phase, roles in recorder.observations
        if phase == InterceptorPhase.BEFORE_ITERATION
    ]

    assert before_iteration_roles == [
        [MessageRole.USER],
        [MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL],
    ]


@pytest.mark.asyncio
async def test_async_engine_rebuilds_runtime_before_condensation_after_tool_resume(
    base_request: ResolvedChatRequest,
) -> None:
    request = base_request.model_copy(deep=True)
    request.system.model = "model-a"
    request.tool_config = ResolvedToolConfig(tools=[_server_tool("echo")])
    mock_llm = get_mock_function_calling_llm(
        [
            [
                ToolSelection(
                    tool_id="tool_1",
                    tool_name="echo",
                    tool_kwargs={"value": "x"},
                )
            ],
            ["done"],
        ]
    )
    llm_component = _make_llm_component(mock_llm)
    llm_component.get_tokenizer.return_value = lambda text: list(text)
    runtime_interceptor = RuntimeModelRequestInterceptor(llm_component)
    condensation_observer = _RuntimeRecordingInterceptor(observations=[])

    await _run_async_engine(
        request,
        mock_llm,
        tool_scheduler=_FakeAsyncToolScheduler(),
        request_interceptors=[runtime_interceptor, condensation_observer],
        llm_component=llm_component,
    )

    before_iteration = [
        observation
        for observation in condensation_observer.observations
        if observation.phase == InterceptorPhase.BEFORE_ITERATION
    ]

    assert len(before_iteration) == 2
    assert all(observation.model_id == "model-a" for observation in before_iteration)
    assert all(
        observation.effective_token_limit is not None
        for observation in before_iteration
    )
    assert all(observation.has_tokenizer for observation in before_iteration)
    assert llm_component.get_tokenizer.call_count == 3


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


async def _source_tool(query: str) -> list[NodeWithScore]:
    """Tool that simulates semantic search returning source documents.
    The shorter_id is a 4-char code that the LLM will reference as [ab12]."""
    node = TextNode(
        text="Paris is the capital of France.",
        id_="doc_paris_001",
        metadata={
            "source_id": "src_paris",
            "artifact_id": "art_paris",
            "shorter_id": "ab12",
        },
    )
    return [NodeWithScore(node=node, score=0.95)]


def _rebuild_source_tool(name: str) -> ToolSpec:
    return ToolSpec.from_defaults(
        name=name, type=name, runtime="server", async_fn=_source_tool
    )


def _server_source_tool(name: str) -> ToolSpec:
    return ToolSpec.from_defaults(
        name=name,
        type=name,
        runtime="server",
        async_fn=_source_tool,
        execution_metadata=build_rebuild_metadata(_rebuild_source_tool, {"name": name}),
    )


class _DocCallRecorder(ChatRequestLoopInterceptor):
    """Records BEFORE_ITERATION calls and document count in context stack."""

    before_iteration_count: int = Field(default=0)
    document_counts: list[int] = Field(default_factory=list)

    async def intercept(self, context: ChatInterceptorContext) -> None:
        if context.phase == InterceptorPhase.BEFORE_ITERATION:
            self.before_iteration_count += 1
            self.document_counts.append(
                len(context.state.input.context_stack.all_documents())
            )


@pytest.mark.asyncio
async def test_document_processing_interceptor_runs_with_documents_on_resume(
    base_request: ResolvedChatRequest,
) -> None:
    """Verify DocumentProcessingRequestInterceptor is called during
    BEFORE_ITERATION on both initial run and after resume, and that
    documents from tool results are available in the context stack."""
    request = base_request.model_copy(deep=True)
    request.citation = CitationConfig(enabled=True)
    request.tool_config = ResolvedToolConfig(tools=[_server_source_tool("search")])

    recorder = _DocCallRecorder()
    citation_interceptor = CitationRequestInterceptor()
    doc_interceptor = DocumentProcessingRequestInterceptor(
        add_context_to_system_prompt=False
    )

    result = await _run_async_engine(
        request,
        get_mock_function_calling_llm(
            [
                [
                    ToolSelection(
                        tool_id="tool_1",
                        tool_name="search",
                        tool_kwargs={"query": "Paris"},
                    )
                ],
                ["Paris is the capital of France."],
            ]
        ),
        tool_scheduler=_FakeAsyncToolScheduler(),
        request_interceptors=[
            citation_interceptor,
            doc_interceptor,
            recorder,
        ],
    )

    # BEFORE_ITERATION must run twice: initial + after resume
    assert recorder.before_iteration_count == 2, (
        f"Expected 2 BEFORE_ITERATION calls, got {recorder.before_iteration_count}"
    )

    # First BEFORE_ITERATION: no tool has run yet → 0 documents
    # Second BEFORE_ITERATION: tool result added sources → docs available
    assert len(recorder.document_counts) == 2
    assert recorder.document_counts[0] == 0, (
        f"Expected 0 docs on first BEFORE_ITERATION, got {recorder.document_counts[0]}"
    )
    assert recorder.document_counts[1] > 0, (
        f"Expected docs on second BEFORE_ITERATION (after resume), "
        f"got {recorder.document_counts[1]}"
    )

    assert result.states[-1].output.status == ChatStatus.COMPLETED


@pytest.mark.asyncio
async def test_extract_citation_interceptor_converts_bracket_refs_on_resume(
    base_request: ResolvedChatRequest,
) -> None:
    """Verify that [XXXX] citations in the LLM output are converted
    to <citation> XML tags by ExtractCitationInterceptor on resume.
    Without this conversion the output leaks internal citation markers."""
    request = base_request.model_copy(deep=True)
    request.citation = CitationConfig(enabled=True)
    request.tool_config = ResolvedToolConfig(tools=[_server_source_tool("search")])

    citation_req_interceptor = CitationRequestInterceptor()
    doc_interceptor = DocumentProcessingRequestInterceptor(
        add_context_to_system_prompt=False
    )
    extract_interceptor = ExtractCitationInterceptor()

    resolved_llm_component = _make_llm_component(
        get_mock_function_calling_llm(
            [
                [
                    ToolSelection(
                        tool_id="tool_1",
                        tool_name="search",
                        tool_kwargs={"query": "Paris"},
                    )
                ],
                ["Paris is the capital of France. [ab12]"],
            ]
        )
    )
    tool_scheduler = _FakeAsyncToolScheduler()
    engine = AsyncChatEngine(
        llm_component=resolved_llm_component,
        request_interceptors=[citation_req_interceptor, doc_interceptor],
        response_interceptors=[extract_interceptor],
        max_iterations=6,
        tool_scheduler=tool_scheduler,
        chat_scheduler=_FakeChatScheduler(),
    )

    all_events: list[Any] = []
    channel = LocalEventChannel()
    state = await engine.execute(request, channel=channel)
    await channel.close()
    all_events.extend(await _drain(channel))

    while state.output.status == ChatStatus.WAITING:
        responses = await tool_scheduler.complete_pending()
        resumed_request = state.input.request.model_copy(deep=True)
        resumed_request.messages = [
            *resumed_request.messages,
            *(response.tool_message for response in responses),
        ]
        channel2 = LocalEventChannel()
        state = await engine.resume(
            AsyncChatCheckpoint(
                checkpoint=state.output.pause_type,
                input=ChatInputState(
                    request=resumed_request,
                    context_stack=state.input.context_stack,
                ),
                iteration=state.runtime.iteration,
                next_block_count=state.runtime.next_block_count,
                payload=IterationCheckpointPayload(
                    model_id=state.runtime.model_id,
                    pending_async_tools=state.output.pending_async_tools,
                    tool_responses=responses,
                    pending_external_tool_calls=state.output.pending_external_tool_calls,
                    total_input_tokens=state.runtime.total_input_tokens,
                    total_output_tokens=state.runtime.total_output_tokens,
                    has_input_usage=state.runtime.has_input_usage,
                    has_output_usage=state.runtime.has_output_usage,
                ),
            ),
            channel=channel2,
        )
        await channel2.close()
        all_events.extend(await _drain(channel2))

    assert state.output.status == ChatStatus.COMPLETED

    # Collect all text deltas that were emitted
    full_text = ""
    for event in all_events:
        if (
            hasattr(event, "delta")
            and event.delta is not None
            and hasattr(event.delta, "text")
        ):
            full_text += event.delta.text or ""

    # The raw [ab12] bracket ref MUST be converted to <citation> tags
    assert "[ab12]" not in full_text, (
        f"Raw citation marker found in output: {full_text!r}"
    )
    assert "<citation" in full_text, (
        f"Expected <citation> XML tag in output, got: {full_text!r}"
    )
