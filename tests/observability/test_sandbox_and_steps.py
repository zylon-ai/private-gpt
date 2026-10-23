"""Sandbox operation spans, spans inside workflow steps, token carry-over."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest
from llama_index.core.workflow import (  # noqa: TC002 - resolved at runtime by @step
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from opentelemetry import trace
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from private_gpt.components.sandbox.base import (
    SandboxExecutionResult,
    SandboxSession,
)
from private_gpt.observability import setup
from private_gpt.observability.processors import SemconvSpanProcessor
from private_gpt.settings.settings import ObservabilitySettings

if TYPE_CHECKING:
    from collections.abc import Iterator

    from private_gpt.components.sandbox.base import SandboxExecOptions

pytestmark = pytest.mark.anyio


@pytest.fixture
def memory() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        SemconvSpanProcessor("openinference", SimpleSpanProcessor(exporter))  # ty: ignore[invalid-argument-type]
    )
    trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    setup._configured = True
    setup._settings = ObservabilitySettings(mode="otel")
    setup._capture_content = True
    HTTPXClientInstrumentor().instrument(tracer_provider=provider)
    try:
        yield exporter
    finally:
        HTTPXClientInstrumentor().uninstrument()
        setup._reset_for_tests()
        trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]


class _HttpSandbox(SandboxSession):
    """A backend that talks HTTP, like OpenSandbox."""

    async def exec(
        self, command: str, opts: SandboxExecOptions | None = None
    ) -> SandboxExecutionResult:
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json={}))
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post("http://sandbox/run")
            await client.get("http://sandbox/status")
        failed = "missing" in command
        return SandboxExecutionResult(
            success=not failed,
            stdout="" if failed else "ok\n",
            stderr="No such file" if failed else "",
            exit_code=1 if failed else 0,
        )

    async def read_file(self, path: str) -> bytes:
        return b"abc"

    async def write_file(self, path: str, content: bytes) -> None: ...

    async def path_exists(self, path: str) -> bool:
        return True

    async def is_dir(self, path: str) -> bool:
        return False

    async def list_dir(self, path: str) -> list[str]:
        return []

    async def make_dir(self, path: str, *a: object, **k: object) -> None: ...

    async def chmod(self, path: str, mode: int) -> None: ...

    async def close(self, force: bool = False) -> None: ...


async def test_sandbox_operations_replace_raw_http_spans(
    memory: InMemorySpanExporter,
) -> None:
    sandbox = _HttpSandbox()
    await sandbox.exec("echo ok")
    await sandbox.exec("cat missing")
    await sandbox.read_file("/data/x.csv")
    await sandbox.close(force=True)

    spans = memory.get_finished_spans()
    names = [s.name for s in spans]
    assert names == [
        "sandbox.run",
        "sandbox.run",
        "sandbox.read_file",
        "sandbox.close",
    ]
    ok, failed = spans[0], spans[1]
    assert ok.attributes["pgpt.sandbox.provider"] == "_HttpSandbox"
    assert "echo ok" in ok.attributes["input.value"]
    assert '"exit_code": 0' in ok.attributes["output.value"]
    assert failed.status.status_code is StatusCode.ERROR
    assert "No such file" in failed.attributes["output.value"]


async def test_spans_in_a_workflow_step_nest_under_the_step(
    memory: InMemorySpanExporter,
) -> None:
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    from private_gpt.observability.tracing import SpanKind, start_span

    class Analysis(Workflow):
        @step
        async def analyze(self, ev: StartEvent) -> StopEvent:
            with start_span("tabular.analysis", SpanKind.TOOL):
                pass
            return StopEvent(result="done")

    provider = trace._TRACER_PROVIDER  # type: ignore[attr-defined]
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument(tracer_provider=provider, use_legacy_callback_handler=False)
    try:
        with start_span("tool tabular_analysis", SpanKind.TOOL):
            await Analysis().run()
    finally:
        instrumentor.uninstrument()

    spans = {s.name: s for s in memory.get_finished_spans()}
    analysis = spans["tabular.analysis"]
    step_span = next(s for n, s in spans.items() if n.endswith("Analysis.analyze"))
    assert analysis.parent is not None
    assert analysis.parent.span_id == step_span.context.span_id


def test_llm_wrapper_gets_token_counts_from_its_inner_call(
    memory: InMemorySpanExporter,
) -> None:
    tracer = trace.get_tracer("t")
    with (
        tracer.start_as_current_span(
            "TritonLLM.astream_chat", attributes={"openinference.span.kind": "LLM"}
        ),
        tracer.start_as_current_span(
            "TritonLLM.astream_complete",
            attributes={
                "openinference.span.kind": "LLM",
                "llm.token_count.prompt": 396,
                "llm.token_count.completion": 34,
            },
        ),
    ):
        pass

    spans = {s.name: s for s in memory.get_finished_spans()}
    chat = spans["TritonLLM.astream_chat"].attributes
    assert chat["llm.token_count.prompt"] == 396
    assert chat["llm.token_count.completion"] == 34


async def test_condensation_span_records_the_decision(
    memory: InMemorySpanExporter,
) -> None:
    from llama_index.core.base.llms.types import ChatMessage, MessageRole

    from private_gpt.components.chat.processors.chat_history.memory.tldr_processor import (
        condense_chat_history,
    )
    from private_gpt.observability.tracing import SpanKind, start_span

    history = [
        ChatMessage(role=MessageRole.USER, content="word " * 50),
        ChatMessage(role=MessageRole.ASSISTANT, content="ok"),
    ]
    with start_span("chat.condensation", SpanKind.CHAIN):
        async for _ in condense_chat_history(
            chat_history=history,
            max_length=10_000,
        ):
            pass

    span = next(s for s in memory.get_finished_spans() if s.name == "chat.condensation")
    attributes = span.attributes or {}
    assert attributes["pgpt.condensation.needed"] is False
    assert attributes["pgpt.condensation.max_tokens"] == 10_000
    assert attributes["pgpt.condensation.messages_before"] == 2
    # Short history: decided from the byte bound, without calling the tokenizer.
    assert attributes["pgpt.condensation.byte_bound"] > 0
