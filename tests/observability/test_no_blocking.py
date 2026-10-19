"""Tracing never blocks requests, whatever the backend does."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from private_gpt.observability.processors import BackgroundSpanProcessor

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan


class _Hanging(SpanExporter):
    """A backend that accepts the connection and never answers."""

    def __init__(self) -> None:
        self.release = threading.Event()

    def export(self, spans: object) -> SpanExportResult:
        self.release.wait(30)
        return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        self.release.set()


class _SlowProcessor:
    """A delegate whose on_end is expensive (attribute normalization...)."""

    def __init__(self, delay: float, sink: list[str]) -> None:
        self.delay = delay
        self.sink = sink

    def on_start(self, span: object, parent_context: object = None) -> None: ...

    def on_end(self, span: ReadableSpan) -> None:
        time.sleep(self.delay)
        self.sink.append(span.name)

    def shutdown(self) -> None: ...

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def test_span_end_does_not_wait_for_processing() -> None:
    sink: list[str] = []
    background = BackgroundSpanProcessor(_SlowProcessor(0.05, sink))
    provider = TracerProvider()
    provider.add_span_processor(background)  # ty: ignore[invalid-argument-type]
    tracer = provider.get_tracer("t")

    start = time.perf_counter()
    for index in range(20):
        with tracer.start_as_current_span(f"s{index}"):
            pass
    elapsed = time.perf_counter() - start

    assert elapsed < 0.1  # 20 x 50 ms of processing happen elsewhere
    assert background.force_flush(5000)
    assert sink == [f"s{index}" for index in range(20)]


def test_full_queue_drops_instead_of_waiting() -> None:
    gate = threading.Event()

    class _Stuck(_SlowProcessor):
        def on_end(self, span: ReadableSpan) -> None:
            gate.wait(10)

    background = BackgroundSpanProcessor(_Stuck(0, []), max_queue_size=5)
    provider = TracerProvider()
    provider.add_span_processor(background)  # ty: ignore[invalid-argument-type]
    tracer = provider.get_tracer("t")

    start = time.perf_counter()
    for index in range(100):
        with tracer.start_as_current_span(f"s{index}"):
            pass
    assert time.perf_counter() - start < 0.1
    assert background.dropped >= 90
    gate.set()


async def test_hanging_backend_does_not_lag_the_event_loop() -> None:
    hanging = _Hanging()
    provider = TracerProvider()
    provider.add_span_processor(
        BackgroundSpanProcessor(  # ty: ignore[invalid-argument-type]
            BatchSpanProcessor(
                hanging,
                max_queue_size=64,
                max_export_batch_size=32,
                schedule_delay_millis=1,
            )
        )
    )
    tracer = provider.get_tracer("t")
    worst = 0.0

    async def ticker() -> None:
        nonlocal worst
        for _ in range(50):
            start = time.perf_counter()
            await asyncio.sleep(0.002)
            worst = max(worst, time.perf_counter() - start - 0.002)

    async def producer() -> None:
        for _ in range(50):
            for _ in range(100):
                with tracer.start_as_current_span("s"):
                    pass
            await asyncio.sleep(0)

    await asyncio.gather(ticker(), producer())
    hanging.release.set()
    assert worst < 0.05


def test_shutdown_is_bounded_with_a_dead_backend() -> None:
    hanging = _Hanging()
    batch = BatchSpanProcessor(hanging, export_timeout_millis=100)
    background = BackgroundSpanProcessor(batch)
    provider = TracerProvider()
    provider.add_span_processor(background)  # ty: ignore[invalid-argument-type]
    with provider.get_tracer("t").start_as_current_span("s"):
        pass

    threading.Timer(0.5, hanging.release.set).start()
    start = time.perf_counter()
    background.shutdown()
    assert time.perf_counter() - start < 5


def test_llm_spans_keep_system_and_recent_messages() -> None:
    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from llama_index.core.llms.mock import MockLLM
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    from private_gpt.observability.llama_index_patch import patch_llama_index_handler

    patch_llama_index_handler(4)
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument(tracer_provider=provider, use_legacy_callback_handler=False)
    try:
        messages = [ChatMessage(role=MessageRole.SYSTEM, content="sys")] + [
            ChatMessage(
                role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                content=f"m{i}",
            )
            for i in range(10)
        ]
        MockLLM(max_tokens=5).chat(messages)
    finally:
        instrumentor.uninstrument()

    llm = next(s for s in memory.get_finished_spans() if s.name.endswith(".chat"))
    attributes = dict(llm.attributes or {})
    contents = [
        attributes[f"llm.input_messages.{i}.message.content"]
        for i in range(5)
        if f"llm.input_messages.{i}.message.content" in attributes
    ]
    assert contents == ["sys", "m6", "m7", "m8", "m9"]
    assert attributes["pgpt.llm.input_messages.omitted"] == 6
    # The full argument dump is not recorded on LLM spans.
    assert "m0" not in str(attributes.get("input.value", ""))


def test_nodes_are_readable_in_spans() -> None:
    import json

    from llama_index.core.schema import NodeWithScore, TextNode

    from private_gpt.observability.llama_index_patch import (
        node_info,
        patch_llama_index_handler,
    )

    patch_llama_index_handler(20)
    from openinference.instrumentation.helpers import safe_json_dumps
    from openinference.instrumentation.llama_index import _handler

    node = TextNode(
        text="x" * 400,
        metadata={"artifact": "handbook", "file_name": "h.txt", "terms": ["a"] * 9},
    )
    info = node_info(NodeWithScore(node=node, score=0.123456))
    assert info is not None
    assert info["type"] == "TextNode"
    assert info["score"] == 0.1235
    assert info["metadata"] == {"artifact": "handbook", "file_name": "h.txt"}
    assert info["chars"] == 400
    assert info["text"].endswith("…")

    encoded = json.loads(safe_json_dumps({"nodes": [node]}, cls=_handler._Encoder))
    assert encoded["nodes"][0]["id"] == node.node_id


async def test_wrapping_stream_llm_records_its_output() -> None:
    """Like Triton: astream_chat consumes the stream of astream_complete."""
    from typing import Any

    from llama_index.core.base.llms.types import (
        ChatMessage,
        ChatResponse,
        CompletionResponse,
        LLMMetadata,
    )
    from llama_index.core.llms.callbacks import (
        llm_chat_callback,
        llm_completion_callback,
    )
    from llama_index.core.llms.custom import CustomLLM
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    from private_gpt.observability.llama_index_patch import patch_llama_index_handler

    class Wrapping(CustomLLM):
        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(model_name="wrapping")

        def complete(self, prompt: str, formatted: bool = False, **kw: Any) -> Any:
            return CompletionResponse(text="hi")

        def stream_complete(
            self, prompt: str, formatted: bool = False, **kw: Any
        ) -> Any:
            raise NotImplementedError

        @llm_completion_callback()
        async def astream_complete(
            self, prompt: str, formatted: bool = False, **kw: Any
        ) -> Any:
            async def gen() -> Any:
                text = ""
                for token in ("Hello", " world"):
                    text += token
                    yield CompletionResponse(text=text, delta=token)

            return gen()

        @llm_chat_callback()
        async def astream_chat(self, messages: Any, **kw: Any) -> Any:
            inner = await self.astream_complete("prompt")

            async def gen() -> Any:
                async for response in inner:
                    yield ChatResponse(
                        message=ChatMessage(role="assistant", content=response.text),
                        delta=response.delta,
                    )

            return gen()

    patch_llama_index_handler(20)
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument(tracer_provider=provider, use_legacy_callback_handler=False)
    try:
        stream = await Wrapping().astream_chat([ChatMessage(role="user", content="q")])
        async for _ in stream:
            pass
    finally:
        instrumentor.uninstrument()

    spans = {s.name: dict(s.attributes or {}) for s in memory.get_finished_spans()}
    chat = spans["Wrapping.astream_chat"]
    assert chat["llm.output_messages.0.message.content"] == "Hello world"
    assert "Hello world" in chat["output.value"]
    assert spans["Wrapping.astream_complete"]["output.value"] == "Hello world"
