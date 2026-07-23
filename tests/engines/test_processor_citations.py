"""Tests for the process_citations function."""

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, cast

import pytest

from private_gpt.components.chat.processors.events.citations.citations import (
    process_citations,
)
from private_gpt.components.engines.citations.types import Document
from private_gpt.components.engines.citations.utils import (
    format_cite,
)
from private_gpt.events.models import (
    Event,
    FatalError,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    TextBlock,
    TextDelta,
)

if TYPE_CHECKING:
    from private_gpt.chat.extensions.citation import ZylonCitation


def create_text_delta(text: str) -> RawContentBlockDeltaEvent:
    return RawContentBlockDeltaEvent(block_id="0", delta=TextDelta(text=text))


def create_document(doc_id: str, text: str, shorter_id: str | None = None) -> Document:
    return Document(
        type="document", id_=doc_id, text=text, document_id="1", shorter_id=shorter_id
    )


async def generate_events(*events: Event) -> AsyncGenerator[Event, None]:
    for event in events:
        yield event


@pytest.mark.asyncio
async def test_process_citations_with_citations() -> None:
    doc = create_document("DOC1", "Test document content")
    documents = [doc]

    events = [
        create_text_delta("This is a test "),
        create_text_delta("with citation [DOC1]"),
    ]

    result: list[Event] = []
    async for event in process_citations(
        generate_events(*events), lambda **kwargs: documents
    ):
        result.append(event)

    assert len(result) == 2
    assert isinstance(result[0], RawContentBlockDeltaEvent)
    assert isinstance(result[1], RawContentBlockDeltaEvent)

    result_0 = result[0]
    result_1 = result[1]
    assert isinstance(result_0.delta, TextDelta)
    assert isinstance(result_1.delta, TextDelta)
    assert result_1.delta.text == f"with citation {format_cite(0, doc, 0)}"
    assert result_1.delta.citations is not None
    assert len(result_1.delta.citations) > 0


@pytest.mark.asyncio
async def test_process_citations_without_documents() -> None:
    events = [
        create_text_delta("This is a test without citations"),
    ]

    result: list[Event] = []
    async for event in process_citations(generate_events(*events), None):
        result.append(event)

    assert len(result) == 1
    assert isinstance(result[0], RawContentBlockDeltaEvent)
    result_0 = cast(RawContentBlockDeltaEvent, result[0])
    assert isinstance(result_0.delta, TextDelta)
    assert result_0.delta.text == "This is a test without citations"
    assert not hasattr(result_0.delta, "citations") or result_0.delta.citations is None


@pytest.mark.asyncio
async def test_process_citations_with_error() -> None:
    error_event = FatalError.from_exception(Exception("Test error"))

    result: list[Event] = []
    async for event in process_citations(generate_events(error_event), None):
        result.append(event)

    assert len(result) == 1
    assert isinstance(result[0], FatalError)
    error = result[0]
    assert error.error.message is not None


@pytest.mark.asyncio
async def test_multiple_citations_processing() -> None:
    doc1 = create_document("DOC1", "First document content")
    doc2 = create_document("DOC2", "Second document content")
    documents = [doc1, doc2]

    events = [
        create_text_delta("Citation one "),
        create_text_delta("[DOC1] and citation two "),
        create_text_delta("[DOC2]"),
    ]

    result: list[Event] = []
    async for event in process_citations(
        generate_events(*events), lambda **kwargs: documents
    ):
        result.append(event)

    assert len(result) == 3
    combined_text = ""
    for e in result:
        if isinstance(e, RawContentBlockDeltaEvent) and isinstance(e.delta, TextDelta):
            combined_text += e.delta.text or ""

    assert format_cite(0, doc1, 0) in combined_text
    assert format_cite(0, doc2, 1) in combined_text

    all_citations: list[ZylonCitation] = []
    for event in result:
        if (
            isinstance(event, RawContentBlockDeltaEvent)
            and isinstance(event.delta, TextDelta)
            and event.delta.citations
        ):
            all_citations.extend(event.delta.citations)

    assert len(all_citations) == 2


@pytest.mark.asyncio
async def test_process_citations_trailing_backtick() -> None:
    doc = create_document("DOC1", "Test document content")
    documents = [doc]

    cb_text = []

    def my_cb(t, c):
        cb_text.append(t)

    # Use fresh event list to avoid mutation reuse
    events = [
        create_text_delta("This is a test with a trailing backtick `"),
    ]
    async for _event in process_citations(
        generate_events(*events), lambda **kwargs: documents, callback=my_cb
    ):
        pass

    assert cb_text[0] == "This is a test with a trailing backtick `"


@pytest.mark.asyncio
async def test_process_citations_trailing_backtick_with_stop_event() -> None:
    doc = create_document("DOC1", "Test document content")
    documents = [doc]

    events = [
        RawContentBlockStartEvent(block_id="0", content_block=TextBlock(text="")),
        create_text_delta("This is a test with a trailing backtick `"),
        RawContentBlockStopEvent(block_id="0"),
    ]

    result: list[Event] = []
    async for event in process_citations(
        generate_events(*events), lambda **kwargs: documents
    ):
        result.append(event)

    # start, delta (without backtick), delta (with backtick), stop
    assert len(result) == 4

    combined = ""
    for r in result:
        if isinstance(r, RawContentBlockDeltaEvent) and isinstance(r.delta, TextDelta):
            combined += r.delta.text or ""

    assert combined == "This is a test with a trailing backtick `"
