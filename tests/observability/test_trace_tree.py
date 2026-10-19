"""A `/v1/messages` turn must export one span tree with the right statuses."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode
from pydantic import Field, PrivateAttr

from private_gpt.chat.input_models import MessageInput
from private_gpt.components.llm.custom.base import ZylonLLM
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.skills.models.skill_entities import SkillFilter
from private_gpt.components.tools.builders.skill_management_builder import (
    SkillManagementToolBuilder,
)
from private_gpt.observability import set_span_status, setup, start_span
from private_gpt.observability.processors import (
    DropSpansProcessor,
    FanOutSpanProcessor,
    SemconvSpanProcessor,
)
from private_gpt.server.chat.chat_router import ChatBody
from private_gpt.server.utils.artifact_input import SkillArtifact
from private_gpt.settings.settings import ObservabilitySettings
from tests.server.chat.test_chat_lifecycle_e2e import _create_skill, _skill_tools

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from httpx import AsyncClient
    from llama_index.core.base.llms.types import (
        ChatResponseAsyncGen,
    )
    from opentelemetry.sdk.trace import ReadableSpan

    from tests.fixtures.mock_injector import MockInjector

pytestmark = pytest.mark.anyio


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        DropSpansProcessor(  # ty: ignore[invalid-argument-type]
            patterns=["*.get_metadata", "*._prepare_chat_with_tools"],
            delegate=SemconvSpanProcessor(
                semconv="openinference",
                delegate=FanOutSpanProcessor([SimpleSpanProcessor(memory)]),
            ),
        )
    )
    # The global provider can only be set once per process; route the proxy
    # tracer to this test provider instead.
    trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    setup._configured = True
    setup._settings = ObservabilitySettings(mode="otel")
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument(tracer_provider=provider, use_legacy_callback_handler=False)
    try:
        yield memory
    finally:
        instrumentor.uninstrument()
        setup._reset_for_tests()
        trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]


class _ScriptedLLM(FunctionCallingLLM, ZylonLLM):
    """Real llama-index LLM (so the dispatcher emits spans) with scripted turns."""

    turns: list[list[str | ToolSelection | BaseException]] = Field(default_factory=list)
    _turn: int = PrivateAttr(default=0)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name="scripted", is_function_calling_model=True, is_chat_model=True
        )

    def get_metadata(self, **kwargs: Any) -> LLMMetadata:
        return self.metadata

    def _prepare_chat_with_tools(
        self,
        tools: Sequence[Any],
        user_msg: Any = None,
        chat_history: list[ChatMessage] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {"messages": list(chat_history or []), "tools": tools, **kwargs}

    def get_tool_calls_from_response(
        self, response: ChatResponse, error_on_no_tool_call: bool = True, **kwargs: Any
    ) -> list[ToolSelection]:
        return response.message.additional_kwargs.get("tool_calls") or []

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        deltas = self.turns[min(self._turn, len(self.turns) - 1)]
        self._turn += 1

        async def gen() -> ChatResponseAsyncGen:
            text = ""
            for delta in deltas:
                if isinstance(delta, BaseException):
                    raise delta
                tool_calls = [delta] if isinstance(delta, ToolSelection) else None
                text += delta if isinstance(delta, str) else ""
                message = ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=text or None,
                    additional_kwargs={"tool_calls": tool_calls},
                )
                yield ChatResponse(
                    message=message,
                    delta=delta if isinstance(delta, str) else None,
                    raw={"usage": {"prompt_tokens": 10, "completion_tokens": 2}},
                    additional_kwargs=message.additional_kwargs,
                )

        return gen()

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        raise NotImplementedError

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Any:
        raise NotImplementedError

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Any:
        raise NotImplementedError

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Any:
        raise NotImplementedError

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Any:
        raise NotImplementedError

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Any:
        raise NotImplementedError


async def test_messages_turn_with_tool_is_one_trace(
    async_test_client: AsyncClient,
    injector: MockInjector,
    exporter: InMemorySpanExporter,
) -> None:
    collection = str(uuid.uuid4())
    await _create_skill(async_test_client, collection=collection, name="trace-probe")
    exporter.clear()
    llm = _ScriptedLLM(
        message_to_input=Mock(),
        completion_to_input=Mock(),
        turns=[
            [
                ToolSelection(
                    tool_id="t1",
                    tool_name="list_skills",
                    tool_kwargs={"page": 0, "page_size": 20},
                )
            ],
            ["done"],
        ],
    )
    llm_component = injector.get(LLMComponent)
    llm_component.get_llm = Mock(return_value=llm)
    injector.bind_mock(LLMComponent, llm_component)
    body = ChatBody(
        messages=[MessageInput(content="hello", role="user")],
        tools=_skill_tools(),
        tool_context=[SkillArtifact(skill_filter=SkillFilter(collection=collection))],
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())
    assert response.status_code == 200, response.text

    spans = exporter.get_finished_spans()
    by_name = {s.name: s for s in spans}
    turn = by_name["chat.turn"]
    assert turn.attributes["openinference.span.kind"] == "AGENT"

    # Every span of the turn's trace reaches chat.turn through its parents
    # (or is an ancestor of it, i.e. the HTTP server span).
    turn_trace = [s for s in spans if s.context.trace_id == turn.context.trace_id]
    ids = {s.context.span_id: s for s in turn_trace}
    ancestors = set()
    node = turn
    while node.parent is not None and node.parent.span_id in ids:
        node = ids[node.parent.span_id]
        ancestors.add(node.context.span_id)
    for span in turn_trace:
        node = span
        while node.context.span_id != turn.context.span_id and node.parent:
            if node.context.span_id in ancestors or node.parent.span_id not in ids:
                break
            node = ids[node.parent.span_id]
        assert (
            node.context.span_id == turn.context.span_id
            or node.context.span_id in ancestors
        ), f"{span.name} is not under chat.turn"
    # Nothing exported for this request lives in another trace.
    orphans = [
        s.name
        for s in spans
        if s.context.trace_id != turn.context.trace_id
        and not s.name.startswith(("POST /v1/skills", "SkillRepository"))
    ]
    assert not orphans, orphans

    iterations = [s for s in turn_trace if s.name.startswith("chat.iteration")]
    assert len(iterations) == 2
    assert all(s.parent.span_id == turn.context.span_id for s in iterations)

    tool = by_name["tool list_skills"]
    assert tool.attributes["openinference.span.kind"] == "TOOL"
    assert tool.attributes["tool.name"] == "list_skills"
    assert tool.parent.span_id == iterations[0].context.span_id

    llm_spans = [
        s for s in turn_trace if s.attributes.get("openinference.span.kind") == "LLM"
    ]
    assert llm_spans, "LLM spans should nest under the iterations"
    iteration_ids = {s.context.span_id for s in iterations}
    assert all(s.parent.span_id in iteration_ids for s in llm_spans)
    for span in llm_spans:
        for key, value in span.attributes.items():
            if key.endswith("tool.json_schema"):
                assert "object at 0x" not in value

    # Session grouping attributes are copied onto every span of the turn.
    assert {s.attributes.get("session.id") for s in [*iterations, tool]} == {
        turn.attributes["session.id"]
    }
    # Noisy leaves are dropped.
    assert not any(s.name.endswith(".get_metadata") for s in spans)


# ---------------------------------------------------------------------------
# Span status: OK on success, ERROR wherever something failed, so a failed
# turn can be found by status alone (Phoenix "status: error" filter).
# ---------------------------------------------------------------------------


def _scripted(*turns: list[str | ToolSelection | BaseException]) -> _ScriptedLLM:
    return _ScriptedLLM(
        message_to_input=Mock(),
        completion_to_input=Mock(),
        turns=list(turns),
    )


def _list_skills_call(tool_id: str = "t1") -> ToolSelection:
    return ToolSelection(
        tool_id=tool_id,
        tool_name="list_skills",
        tool_kwargs={"page": 0, "page_size": 20},
    )


async def _run_turn(
    client: AsyncClient,
    injector: MockInjector,
    exporter: InMemorySpanExporter,
    llm: _ScriptedLLM,
) -> tuple[int, dict[str, ReadableSpan], list[ReadableSpan]]:
    collection = str(uuid.uuid4())
    await _create_skill(client, collection=collection, name="status-probe")
    exporter.clear()
    llm_component = injector.get(LLMComponent)
    llm_component.get_llm = Mock(return_value=llm)
    injector.bind_mock(LLMComponent, llm_component)
    body = ChatBody(
        messages=[MessageInput(content="hello", role="user")],
        tools=_skill_tools(),
        tool_context=[SkillArtifact(skill_filter=SkillFilter(collection=collection))],
    )
    response = await client.post("/v1/messages", json=body.model_dump())
    spans = list(exporter.get_finished_spans())
    by_name = {s.name: s for s in spans}
    turn = by_name["chat.turn"]
    turn_trace = [s for s in spans if s.context.trace_id == turn.context.trace_id]
    return response.status_code, by_name, turn_trace


def _status(span: ReadableSpan) -> StatusCode:
    return span.status.status_code


def _ours(spans: list[ReadableSpan]) -> list[ReadableSpan]:
    """Spans created by private_gpt (turn, iterations, tools)."""
    return [s for s in spans if s.name.startswith(("chat.", "tool "))]


async def test_successful_turn_marks_every_span_ok(
    async_test_client: AsyncClient,
    injector: MockInjector,
    exporter: InMemorySpanExporter,
) -> None:
    _, by_name, turn_trace = await _run_turn(
        async_test_client,
        injector,
        exporter,
        _scripted([_list_skills_call()], ["done"]),
    )

    ours = _ours(turn_trace)
    assert {s.name for s in ours} >= {
        "chat.turn",
        "chat.iteration 1",
        "chat.iteration 2",
        "tool list_skills",
    }
    assert {s.name: _status(s) for s in ours} == {s.name: StatusCode.OK for s in ours}
    assert by_name["tool list_skills"].attributes["pgpt.tool.outcome"] == "success"
    # No span in the turn is marked failed, LLM spans included.
    assert not [s.name for s in turn_trace if _status(s) == StatusCode.ERROR]


async def test_llm_stream_failure_marks_iteration_and_turn_error(
    async_test_client: AsyncClient,
    injector: MockInjector,
    exporter: InMemorySpanExporter,
) -> None:
    _, by_name, turn_trace = await _run_turn(
        async_test_client,
        injector,
        exporter,
        _scripted(["partial ", RuntimeError("triton stream broke")]),
    )

    for name in ("chat.turn", "chat.iteration 1"):
        span = by_name[name]
        assert _status(span) == StatusCode.ERROR, name
        assert span.status.description == "RuntimeError: triton stream broke"
        assert any(e.name == "exception" for e in span.events), name
    # The LLM span itself reports the failure too (OpenInference instrumentor).
    llm_spans = [
        s for s in turn_trace if s.attributes.get("openinference.span.kind") == "LLM"
    ]
    assert any(_status(s) == StatusCode.ERROR for s in llm_spans)


async def test_failure_in_second_iteration_keeps_first_ok(
    async_test_client: AsyncClient,
    injector: MockInjector,
    exporter: InMemorySpanExporter,
) -> None:
    _, by_name, _ = await _run_turn(
        async_test_client,
        injector,
        exporter,
        _scripted([_list_skills_call()], [RuntimeError("second call failed")]),
    )

    assert _status(by_name["chat.iteration 1"]) == StatusCode.OK
    assert _status(by_name["tool list_skills"]) == StatusCode.OK
    assert _status(by_name["chat.iteration 2"]) == StatusCode.ERROR
    assert _status(by_name["chat.turn"]) == StatusCode.ERROR


async def test_tool_exception_marks_tool_error_and_turn_still_ok(
    async_test_client: AsyncClient,
    injector: MockInjector,
    exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(*_: Any, **__: Any) -> Any:
        raise ConnectionError("skills database unreachable")

    # Only the tool's own lookup fails; request setup (which also lists
    # skills) keeps working.
    monkeypatch.setattr(SkillManagementToolBuilder, "_resolved", boom)
    status_code, by_name, _ = await _run_turn(
        async_test_client,
        injector,
        exporter,
        _scripted([_list_skills_call()], ["sorry, the tool failed"]),
    )

    # The error goes back to the model as a tool result and the turn recovers:
    # only the tool span is a failure.
    assert status_code == 200
    tool = by_name["tool list_skills"]
    assert _status(tool) == StatusCode.ERROR
    assert "skills database unreachable" in (tool.status.description or "")
    assert tool.attributes["pgpt.tool.outcome"] == "failure"
    assert _status(by_name["chat.iteration 1"]) == StatusCode.OK
    assert _status(by_name["chat.iteration 2"]) == StatusCode.OK
    assert _status(by_name["chat.turn"]) == StatusCode.OK


async def test_unknown_tool_marks_tool_error(
    async_test_client: AsyncClient,
    injector: MockInjector,
    exporter: InMemorySpanExporter,
) -> None:
    _, by_name, _ = await _run_turn(
        async_test_client,
        injector,
        exporter,
        _scripted(
            [ToolSelection(tool_id="t1", tool_name="does_not_exist", tool_kwargs={})],
            ["ok"],
        ),
    )

    tool = by_name["tool does_not_exist"]
    assert _status(tool) == StatusCode.ERROR
    assert tool.status.description == "Tool 'does_not_exist' not found."
    assert tool.attributes["pgpt.tool.status"] == "not_found"
    assert _status(by_name["chat.turn"]) == StatusCode.OK


# ---------------------------------------------------------------------------
# start_span / set_span_status contract
# ---------------------------------------------------------------------------


def test_span_closes_ok_by_default(exporter: InMemorySpanExporter) -> None:
    with start_span("work"):
        pass
    (span,) = exporter.get_finished_spans()
    assert _status(span) == StatusCode.OK


def test_exception_marks_span_error_and_is_reraised(
    exporter: InMemorySpanExporter,
) -> None:
    with pytest.raises(ValueError, match="bad input"), start_span("work"):
        raise ValueError("bad input")
    (span,) = exporter.get_finished_spans()
    assert _status(span) == StatusCode.ERROR
    assert span.status.description == "ValueError: bad input"
    assert [e.name for e in span.events] == ["exception"]


def test_error_outcome_is_not_overwritten_on_close(
    exporter: InMemorySpanExporter,
) -> None:
    with start_span("tool x") as span:
        set_span_status(span, error="returned an error result")
    (finished,) = exporter.get_finished_spans()
    assert _status(finished) == StatusCode.ERROR
    assert finished.status.description == "returned an error result"


def test_cancellation_is_not_an_error(exporter: InMemorySpanExporter) -> None:
    import asyncio

    with pytest.raises(asyncio.CancelledError), start_span("turn"):
        raise asyncio.CancelledError
    (span,) = exporter.get_finished_spans()
    assert _status(span) == StatusCode.UNSET
    assert span.attributes["pgpt.cancelled"] is True
