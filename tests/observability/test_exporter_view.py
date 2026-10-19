"""Per-exporter view: chat.turn as root, and folded user input."""

# ruff: noqa: SIM117 - nesting is the parent/child structure under test

from __future__ import annotations

import base64

from llama_index.core.base.llms.types import (
    AudioBlock,
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind, Status, StatusCode

from private_gpt.observability.chat import user_input_attributes
from private_gpt.observability.exporters import TURN_ROUTES
from private_gpt.observability.message import last_user_message
from private_gpt.observability.processors import ExporterViewProcessor

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _provider(memory: InMemorySpanExporter, **kwargs: object) -> TracerProvider:
    provider = TracerProvider()
    provider.add_span_processor(
        ExporterViewProcessor(  # ty: ignore[invalid-argument-type]
            SimpleSpanProcessor(memory),
            turn_root=True,
            turn_routes=TURN_ROUTES,
            **kwargs,  # type: ignore[arg-type]
        )
    )
    return provider


def test_turn_becomes_root_and_http_plumbing_is_hidden() -> None:
    memory = InMemorySpanExporter()
    tracer = _provider(memory).get_tracer("t")
    with tracer.start_as_current_span("POST /v1/messages", kind=SpanKind.SERVER):
        with tracer.start_as_current_span("ChatRequestMapper.map"):
            with tracer.start_as_current_span("GET auth"):
                pass
        with tracer.start_as_current_span("chat.turn"):
            with tracer.start_as_current_span("chat.iteration 1"):
                pass

    spans = {span.name: span for span in memory.get_finished_spans()}
    assert set(spans) == {"chat.turn", "chat.iteration 1"}
    assert spans["chat.turn"].parent is None
    assert spans["chat.iteration 1"].parent is not None
    assert (
        spans["chat.iteration 1"].parent.span_id == spans["chat.turn"].context.span_id
    )


def test_async_turn_after_response_is_still_a_root() -> None:
    memory = InMemorySpanExporter()
    tracer = _provider(memory).get_tracer("t")
    with tracer.start_as_current_span(
        "POST /v1/messages/async", kind=SpanKind.SERVER
    ) as http:
        from opentelemetry import context, trace

        ctx = trace.set_span_in_context(http)
    # The turn runs in a background task after the 200 acknowledgement.
    token = context.attach(ctx)
    try:
        with tracer.start_as_current_span("chat.turn"):
            pass
    finally:
        context.detach(token)

    assert [(s.name, s.parent) for s in memory.get_finished_spans()] == [
        ("chat.turn", None)
    ]


def test_http_failure_before_a_turn_is_kept() -> None:
    memory = InMemorySpanExporter()
    tracer = _provider(memory).get_tracer("t")
    with tracer.start_as_current_span(
        "POST /v1/messages", kind=SpanKind.SERVER
    ) as span:
        span.set_status(Status(StatusCode.ERROR, "HTTP 400"))

    assert [s.name for s in memory.get_finished_spans()] == ["POST /v1/messages"]


def test_other_routes_and_resource_are_untouched_or_per_exporter() -> None:
    memory = InMemorySpanExporter()
    tracer = _provider(
        memory, resource={"openinference.project.name": "chat"}
    ).get_tracer("t")
    with tracer.start_as_current_span(
        "POST /v1/tools/web-search", kind=SpanKind.SERVER
    ):
        with tracer.start_as_current_span("web_search.query"):
            pass

    spans = memory.get_finished_spans()
    assert sorted(s.name for s in spans) == [
        "POST /v1/tools/web-search",
        "web_search.query",
    ]
    assert all(
        s.resource.attributes["openinference.project.name"] == "chat" for s in spans
    )


def test_folded_multimodal_user_input() -> None:
    messages = [
        ChatMessage(role=MessageRole.USER, content="first question"),
        ChatMessage(role=MessageRole.ASSISTANT, content="answer"),
        ChatMessage(
            role=MessageRole.USER,
            blocks=[
                TextBlock(text="what is in these?"),
                ImageBlock(image=PNG, image_mimetype="image/png"),
                AudioBlock(audio=b"\x00RIFF\xff0000WAVE", format="audio/wav"),
            ],
            additional_kwargs={"document": []},
        ),
        ChatMessage(role=MessageRole.TOOL, content="tool output"),
    ]

    folded = last_user_message(messages)
    assert folded is not None
    types = [block["type"] for block in folded["content"]]
    assert types == ["text", "image", "audio"]
    image = folded["content"][1]
    assert image["url"].startswith("data:image/png;base64,")
    assert folded["content"][2]["media_type"] == "audio/wav"

    attributes = user_input_attributes(messages)
    text, image_md, audio = attributes["input"].split("\n\n")
    assert text == "what is in these?"
    assert image_md.startswith("![image/png](data:image/png;base64,")
    assert audio == "[audio: audio/wav, 14 B]"
    prefix = "llm.input_messages.0.message.contents"
    assert attributes[f"{prefix}.1.message_content.type"] == "image"


def test_folded_document_input() -> None:
    from private_gpt.events.models._content_blocks import DocumentBlock

    document = DocumentBlock(
        source=DocumentBlock.PlainTextSource(
            data="The policy says 23 days.", media_type="text/plain"
        ),
        title="policy.txt",
    )
    folded = last_user_message(
        [
            ChatMessage(
                role=MessageRole.USER,
                content="summarize",
                additional_kwargs={"document": [document]},
            )
        ]
    )
    assert folded is not None
    assert folded["content"][1]["type"] == "document"
    assert folded["content"][1]["title"] == "policy.txt"
    assert folded["content"][1]["source"]["text"] == "The policy says 23 days."
