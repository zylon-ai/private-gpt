import asyncio
from collections.abc import AsyncGenerator

import pytest

from private_gpt.components.chat.processors.events.citations.citations import (
    process_citations,
)
from private_gpt.components.engines.citations.types import Document
from private_gpt.components.engines.citations.utils import format_cite
from private_gpt.events.models import (
    Event,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
)


def create_document(citation_id: str) -> Document:
    return Document(
        type="document",
        id_=f"source-{citation_id}",
        shorter_id=citation_id,
        document_id=f"artifact-{citation_id}",
        text=citation_id,
    )


async def delayed_events(
    events: list[Event], delay_seconds: float = 0
) -> AsyncGenerator[Event, None]:
    for event in events:
        await asyncio.sleep(delay_seconds)
        yield event


async def collect_text_stream(
    chunks: list[str],
    documents: list[Document],
    delay_seconds: float = 0,
) -> tuple[str, int]:
    events: list[Event] = [
        RawContentBlockStartEvent(block_id="text", content_block=TextBlock(text="")),
        *[
            RawContentBlockDeltaEvent(
                block_id="text",
                delta=TextDelta(text=chunk),
            )
            for chunk in chunks
        ],
        RawContentBlockStopEvent(block_id="text"),
    ]
    output = ""
    citation_count = 0

    async for event in process_citations(
        delayed_events(events, delay_seconds),
        lambda **kwargs: documents,
    ):
        if not isinstance(event, RawContentBlockDeltaEvent):
            continue
        if not isinstance(event.delta, TextDelta):
            continue
        output += event.delta.text or ""
        citation_count += len(event.delta.citations or [])

    return output, citation_count


async def collect_thinking_stream(
    chunks: list[str], documents: list[Document]
) -> tuple[str, int]:
    events: list[Event] = [
        RawContentBlockStartEvent(
            block_id="thinking",
            content_block=ThinkingBlock(thinking="", signature=""),
        ),
        *[
            RawContentBlockDeltaEvent(
                block_id="thinking",
                delta=ThinkingDelta(thinking=chunk),
            )
            for chunk in chunks
        ],
        RawContentBlockStopEvent(block_id="thinking"),
    ]
    output = ""
    citation_count = 0

    async for event in process_citations(
        delayed_events(events),
        lambda **kwargs: documents,
    ):
        if not isinstance(event, RawContentBlockDeltaEvent):
            continue
        if not isinstance(event.delta, ThinkingDelta):
            continue
        output += event.delta.thinking or ""
        citation_count += len(event.delta.citations or [])

    return output, citation_count


@pytest.mark.asyncio
async def test_every_event_split_inside_wrapped_citation_is_stable() -> None:
    document = create_document("AB12")
    text = "Before `[AB12]` after."
    expected = f"Before {format_cite(0, document, 0)} after."

    for split_at in range(len(text) + 1):
        output, citation_count = await collect_text_stream(
            [text[:split_at], text[split_at:]],
            [document],
        )
        assert output == expected, f"split_at={split_at}"
        assert citation_count == 1, f"split_at={split_at}"


@pytest.mark.asyncio
async def test_delayed_character_stream_preserves_output_and_citations() -> None:
    first = create_document("AB12")
    second = create_document("CD34")
    text = "Before `[AB12], [CD34]` after."

    output, citation_count = await collect_text_stream(
        list(text),
        [first, second],
        delay_seconds=0.0001,
    )

    assert output == (
        f"Before {format_cite(0, first, 0)}, {format_cite(1, second, 1)} after."
    )
    assert citation_count == 2


@pytest.mark.asyncio
async def test_thinking_stream_uses_same_citation_semantics() -> None:
    document = create_document("AB12")

    output, citation_count = await collect_thinking_stream(
        ["Reasoning `", "[AB", "12]", "` complete."],
        [document],
    )

    assert output == f"Reasoning {format_cite(0, document, 0)} complete."
    assert citation_count == 1


@pytest.mark.asyncio
async def test_llm_garbage_around_valid_citation_is_not_lost() -> None:
    document = create_document("AB12")
    chunks = [
        "Trash `[[AB12]]`, `(AB12)`, [UNKNOWN]. ",
        "Valid `[AB",
        "12]` end.",
    ]

    output, citation_count = await collect_text_stream(chunks, [document])

    assert output == (
        f"Trash {format_cite(0, document, 0)}, `(AB12)`, [UNKNOWN]. "
        f"Valid {format_cite(0, document, 0)} end."
    )
    assert citation_count == 2


@pytest.mark.asyncio
async def test_repeated_citation_across_delayed_events_reuses_index() -> None:
    document = create_document("AB12")

    output, citation_count = await collect_text_stream(
        ["First [AB12]", ", second ", "[AB12]."],
        [document],
        delay_seconds=0.0001,
    )

    assert output == (
        f"First {format_cite(0, document, 0)}, second {format_cite(1, document, 0)}."
    )
    assert citation_count == 2
