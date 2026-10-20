"""A `/v1/messages` turn must export one span tree with the right statuses."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

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
            for index, delta in enumerate(deltas):
                if isinstance(delta, BaseException):
                    raise delta
                tool_calls = [delta] if isinstance(delta, ToolSelection) else None
                text += delta if isinstance(delta, str) else ""
                extra: dict[str, Any] = {"tool_calls": tool_calls}
                if index == len(deltas) - 1 and not tool_calls:
                    extra["stop_reason"] = "end_turn"  # as real providers report
                message = ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=text or None,
                    additional_kwargs=extra,
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
# turn can be found by status alone (a "status: error" filter).
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
    **body_fields: Any,
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
    payload = {**body.model_dump(), **body_fields}
    response = await client.post("/v1/messages", json=payload)
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


# ---------------------------------------------------------------------------
# Turn output and sessions
# ---------------------------------------------------------------------------


async def test_turn_output_is_answer_text_with_folded_message(
    async_test_client: AsyncClient,
    injector: MockInjector,
    exporter: InMemorySpanExporter,
) -> None:
    import json

    _, by_name, _ = await _run_turn(
        async_test_client,
        injector,
        exporter,
        _scripted([_list_skills_call()], ["The answer ", "is 42."]),
    )

    turn = by_name["chat.turn"]
    # Session views render output.value as the assistant bubble.
    assert turn.attributes["output.value"] == "The answer is 42."
    assert turn.attributes["output.mime_type"] == "text/plain"
    message = json.loads(turn.attributes["pgpt.output.message"])
    assert message["role"] == "assistant"
    assert message["stop_reason"] == "end_turn"
    types = [block["type"] for block in message["content"]]
    assert types[-1] == "text"
    assert "server_tool_use" in types or "tool_use" in types
    tool_use = next(b for b in message["content"] if b.get("name") == "list_skills")
    assert tool_use["input"] == {"page": 0, "page_size": 20}
    # Wire-only fields are left out.
    assert not any("start_timestamp" in block for block in message["content"])
    assert list(turn.attributes["pgpt.output.tool_calls"]) == ["list_skills"]


async def test_session_is_container_and_user_is_caller_metadata(
    async_test_client: AsyncClient,
    injector: MockInjector,
    exporter: InMemorySpanExporter,
) -> None:
    _, by_name, turn_trace = await _run_turn(
        async_test_client,
        injector,
        exporter,
        _scripted(["hi"]),
        container="thread-123",
        metadata={"user_id": "member-7"},
    )

    assert by_name["chat.turn"].attributes["session.id"] == "thread-123"
    assert by_name["chat.turn"].attributes["user.id"] == "member-7"
    # Every span of the turn carries the same session for grouping.
    assert {s.attributes.get("session.id") for s in _ours(turn_trace)} == {"thread-123"}


async def test_without_container_each_turn_is_its_own_session_and_no_user(
    async_test_client: AsyncClient,
    injector: MockInjector,
    exporter: InMemorySpanExporter,
) -> None:
    _, first, _ = await _run_turn(
        async_test_client, injector, exporter, _scripted(["one"])
    )
    _, second, _ = await _run_turn(
        async_test_client, injector, exporter, _scripted(["two"])
    )

    a, b = first["chat.turn"], second["chat.turn"]
    assert a.attributes["session.id"] != b.attributes["session.id"]
    # The request mapper generates a uuid user_id when none is sent; it must
    # not show up as a user.
    assert "user.id" not in a.attributes
    assert "user.id" not in b.attributes


async def test_follow_up_turns_share_the_session(
    async_test_client: AsyncClient,
    injector: MockInjector,
    exporter: InMemorySpanExporter,
) -> None:
    _, first, _ = await _run_turn(
        async_test_client,
        injector,
        exporter,
        _scripted(["first answer"]),
        container="thread-9",
    )
    _, second, _ = await _run_turn(
        async_test_client,
        injector,
        exporter,
        _scripted(["second answer"]),
        container="thread-9",
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "and then?"},
        ],
    )

    a, b = first["chat.turn"], second["chat.turn"]
    assert a.attributes["session.id"] == b.attributes["session.id"] == "thread-9"
    assert a.context.trace_id != b.context.trace_id
    assert b.attributes["input.value"] == "and then?"
    assert b.attributes["output.value"] == "second answer"


# ---------------------------------------------------------------------------
# Span plumbing
# ---------------------------------------------------------------------------


def test_dropped_span_children_are_reattached_to_its_parent() -> None:
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        DropSpansProcessor(  # ty: ignore[invalid-argument-type]
            patterns=["FunctionTool.acall"],
            delegate=SimpleSpanProcessor(memory),
        )
    )
    tracer = provider.get_tracer("t")
    with (
        tracer.start_as_current_span("tool x") as tool,
        tracer.start_as_current_span("FunctionTool.acall"),
        tracer.start_as_current_span("sandbox.exec"),
    ):
        pass

    spans = {s.name: s for s in memory.get_finished_spans()}
    assert set(spans) == {"tool x", "sandbox.exec"}
    assert spans["sandbox.exec"].parent.span_id == tool.get_span_context().span_id


async def test_traced_records_input_output_and_failure(
    exporter: InMemorySpanExporter,
) -> None:
    from private_gpt.observability import traced

    @traced("step.ok", capture=lambda x: {"x": x}, result=lambda r: {"double": r})
    async def ok(x: int) -> int:
        return x * 2

    @traced("step.fail")
    async def fail() -> None:
        raise RuntimeError("nope")

    assert await ok(21) == 42
    with pytest.raises(RuntimeError):
        await fail()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["step.ok"].attributes["input.value"] == '{"x": 21}'
    assert spans["step.ok"].attributes["output.value"] == '{"double": 42}'
    assert _status(spans["step.ok"]) == StatusCode.OK
    assert _status(spans["step.fail"]) == StatusCode.ERROR


async def test_traced_is_a_passthrough_when_tracing_is_off() -> None:
    from private_gpt.observability import traced

    calls = []

    @traced("step", capture=lambda *_: calls.append("capture"))
    async def work() -> str:
        return "done"

    assert await work() == "done"
    assert calls == []


# ---------------------------------------------------------------------------
# Server tool internals
# ---------------------------------------------------------------------------


async def test_web_search_traces_provider_query_and_processing(
    exporter: InMemorySpanExporter,
) -> None:
    from private_gpt.components.web.web_search.models import WebSearchResult
    from private_gpt.components.web.web_search.web_search_service import (
        WebSearchService,
    )

    hits = [WebSearchResult(title="Rust 1.99", url="https://rust.dev", description="d")]
    service = WebSearchService.__new__(WebSearchService)
    service._initialized = True
    service._settings = Mock(web_search=Mock(num_links=3))
    service._provider = Mock(make_query=_async(hits))
    service._processor = Mock(process_results=_async(hits))
    service.validate = _async(None)

    assert await service.search("rust version") == hits

    spans = {s.name.split(" ")[0]: s for s in exporter.get_finished_spans()}
    query, process = spans["web_search.query"], spans["web_search.process"]
    assert query.attributes["openinference.span.kind"] == "RETRIEVER"
    assert query.attributes["input.value"] == "rust version"
    assert query.attributes["retrieval.documents.0.document.id"] == "https://rust.dev"
    assert query.attributes["pgpt.web_search.results"] == 1
    assert "Rust 1.99" in process.attributes["output.value"]


async def test_sandbox_operations_are_traced_and_failures_marked(
    exporter: InMemorySpanExporter,
) -> None:
    from private_gpt.components.code_execution.results import BashExecutionResult
    from private_gpt.components.code_execution.sandbox_session import (
        SandboxCodeExecutionSession,
    )

    env = Mock(
        id="s1",
        workspace="/home/agent/",
        exec=_async(
            Mock(
                success=True,
                stdout="hello",
                stderr="",
                exit_code=0,
                execution_time_ms=5,
            )
        ),
        touch=Mock(),
    )
    env.sandbox = Mock(path_exists=_async(False))
    session = SandboxCodeExecutionSession(env)

    result = await session.execute_bash("echo hello")
    assert isinstance(result, BashExecutionResult)
    missing = await session.view("missing.txt")
    assert not missing.success

    spans = {s.name: s for s in exporter.get_finished_spans()}
    exec_span = spans["sandbox.exec"]
    assert '"command": "echo hello"' in exec_span.attributes["input.value"]
    assert '"stdout": "hello"' in exec_span.attributes["output.value"]
    assert _status(exec_span) == StatusCode.OK
    view_span = spans["sandbox.view"]
    assert _status(view_span) == StatusCode.ERROR
    assert "File not found" in view_span.status.description


async def test_file_tools_trace_one_span_per_path(
    exporter: InMemorySpanExporter,
) -> None:
    from private_gpt.components.tools.builders.sandbox_file_tools import (
        run_over_paths,
    )

    async def worker(path: str, _lock: Any) -> str:
        if path == "bad.pdf":
            raise ValueError("not a document")
        return f"converted {path}"

    await run_over_paths(
        ["a.pdf", "bad.pdf"],
        worker,
        2,
        gerund="converting",
        past="converted",
        noun="document",
    )

    spans = [
        s for s in exporter.get_finished_spans() if s.name == "converting document"
    ]
    by_path = {s.attributes["pgpt.file.path"]: s for s in spans}
    assert _status(by_path["a.pdf"]) == StatusCode.OK
    assert by_path["a.pdf"].attributes["output.value"] == "converted a.pdf"
    assert _status(by_path["bad.pdf"]) == StatusCode.ERROR


def _async(value: Any) -> Any:
    async def fn(*_: Any, **__: Any) -> Any:
        return value

    return fn


# ---------------------------------------------------------------------------
# Interceptor phases inside the iteration
# ---------------------------------------------------------------------------


async def test_interceptors_that_change_the_request_nest_under_iteration(
    async_test_client: AsyncClient,
    injector: MockInjector,
    exporter: InMemorySpanExporter,
) -> None:
    _, by_name, turn_trace = await _run_turn(
        async_test_client, injector, exporter, _scripted(["hi"])
    )

    interceptors = [s for s in turn_trace if s.name.startswith("interceptor ")]
    assert interceptors, "the system prompt interceptor always changes the request"
    iteration = by_name["chat.iteration 1"]
    in_iteration = [
        s for s in interceptors if s.parent.span_id == iteration.context.span_id
    ]
    assert in_iteration
    for span in interceptors:
        assert span.attributes["pgpt.interceptor.changed"] is True
        assert span.attributes["pgpt.interceptor.phase"]
        assert _status(span) == StatusCode.OK


def test_idle_interceptor_spans_are_dropped_but_failures_kept() -> None:
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        DropSpansProcessor(  # ty: ignore[invalid-argument-type]
            patterns=[], delegate=SimpleSpanProcessor(memory)
        )
    )
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("chat.iteration 1"):
        with tracer.start_as_current_span("interceptor Idle") as idle:
            idle.set_attribute("pgpt.interceptor.changed", False)
        with tracer.start_as_current_span("interceptor Busy") as busy:
            busy.set_attribute("pgpt.interceptor.changed", False)
            with tracer.start_as_current_span("TritonLLM.astream_chat"):
                pass
        with tracer.start_as_current_span("interceptor Broken") as broken:
            broken.set_attribute("pgpt.interceptor.changed", False)
            broken.set_status(StatusCode.ERROR, "boom")

    names = {s.name for s in memory.get_finished_spans()}
    assert "interceptor Idle" not in names
    # Did work (an LLM call, e.g. condensation) or failed: kept.
    assert {"interceptor Busy", "interceptor Broken"} <= names


async def test_describe_image_is_traced_with_image_metadata(
    exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llama_index.core.base.llms.types import ImageBlock

    from private_gpt.components.multimodality import image_handler

    class _Workflow:
        def __init__(self, *_: Any, **__: Any) -> None: ...

        async def run(self, **_: Any) -> Any:
            return Mock(description="a red square")

    monkeypatch.setattr(image_handler, "ImageProcessingWorkflow", _Workflow)
    block = ImageBlock(image=b"\x89PNG", image_mimetype="image/png")

    assert await image_handler.describe_image(Mock(), [block], "what is it?") == (
        "a red square"
    )

    (span,) = [s for s in exporter.get_finished_spans() if s.name == "describe_image"]
    assert '"images": 1' in span.attributes["input.value"]
    assert '"image/png"' in span.attributes["input.value"]
    assert span.attributes["output.value"] == "a red square"
    assert _status(span) == StatusCode.OK


def test_state_snapshot_keeps_conversation_tools_and_params() -> None:
    from llama_index.core.base.llms.types import ChatMessage
    from llama_index.core.llms.llm import ToolSelection

    from private_gpt.observability.chat import state_snapshot

    class _Obj:
        def __init__(self, **kw: object) -> None:
            self.__dict__.update(kw)

    class _Params:
        def model_dump(self, **_: object) -> dict:
            return {"temperature": 0.1}

    state = _Obj(
        input=_Obj(
            request=_Obj(
                system=_Obj(prompt="Be brief."),
                thinking=_Obj(enabled=True, type="low"),
                messages=[
                    ChatMessage(role="user", content="weather?"),
                    ChatMessage(
                        role="assistant",
                        content="",
                        additional_kwargs={
                            "thinking": "Use the tool.",
                            "tool_calls": [
                                ToolSelection(
                                    tool_id="t1",
                                    tool_name="get_weather",
                                    tool_kwargs={"city": "Madrid"},
                                )
                            ],
                        },
                    ),
                    ChatMessage(
                        role="tool",
                        content="sunny",
                        additional_kwargs={
                            "tool_call_id": "t1",
                            "tool_call_name": "get_weather",
                        },
                    ),
                ],
            ),
            context_stack=_Obj(
                all_tools=lambda: [
                    _Obj(name="get_weather", type=None, runtime="server")
                ]
            ),
            llm_kwargs=_Params(),
        ),
        runtime=_Obj(iteration=2, total_input_tokens=100, total_output_tokens=20),
        output=_Obj(
            status="running",
            stop_reason="tool_use",
            pending_external_tool_calls=[],
            pending_async_tools={},
        ),
    )

    assert state_snapshot(state) == {
        "iteration": 2,
        "status": "running",
        "stop_reason": "tool_use",
        "system": "Be brief.",
        "thinking": "low",
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning": "Use the tool.",
                "tool_calls": [
                    {"id": "t1", "name": "get_weather", "arguments": {"city": "Madrid"}}
                ],
            },
            {
                "role": "tool",
                "content": "sunny",
                "tool_call_id": "t1",
                "name": "get_weather",
            },
        ],
        "tools": [{"name": "get_weather", "runtime": "server"}],
        "llm_params": {"temperature": 0.1},
        "tokens": {"input": 100, "output": 20},
    }


def test_in_place_interceptor_changes_are_detected() -> None:
    from llama_index.core.base.llms.types import ChatMessage

    from private_gpt.observability.chat import (
        record_interceptor_result,
        state_fingerprint,
    )

    class _Obj:
        def __init__(self, **kw: object) -> None:
            self.__dict__.update(kw)

    class _Span:
        def __init__(self) -> None:
            self.attributes: dict = {}

        def is_recording(self) -> bool:
            return True

        def set_attributes(self, attributes: dict) -> None:
            self.attributes.update(attributes)

    message = ChatMessage(role="assistant", content="", additional_kwargs={"a": 1})
    request = _Obj(messages=[message], system=_Obj(prompt="p"))
    state = _Obj(input=_Obj(request=request, context_stack=_Obj(layers=[])))

    def run(mutate: object) -> dict:
        span = _Span()
        with patch("private_gpt.observability.chat.is_enabled", return_value=True):
            before = state_fingerprint(state)
            mutate()  # type: ignore[operator]
            record_interceptor_result(span, before, state)
        return span.attributes

    untouched = run(lambda: None)
    assert untouched["pgpt.interceptor.changed"] is False
    # Same message object, kwargs dropped in place (flattening tool calls).
    changed = run(lambda: message.additional_kwargs.pop("a"))
    assert changed["pgpt.interceptor.changed"] is True
    assert changed["pgpt.interceptor.changes"] == ["messages"]
    prompt = run(lambda: setattr(request.system, "prompt", "q"))
    assert prompt["pgpt.interceptor.changes"] == ["system_prompt"]
