import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING

from private_gpt.components.engines.citations.types import Document
from private_gpt.components.engines.citations.utils import (
    extract_citations_by_original_text,
)
from private_gpt.events.models import (
    Event,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    TextDelta,
    ThinkingDelta,
)

if TYPE_CHECKING:
    from private_gpt.components.engines.citations.types import Citation


async def process_citations(
    event_generator: AsyncGenerator[Event, None],
    documents_fn: Callable[..., list[Document]] | None = None,
    citation_indices_fn: Callable[..., dict[str, int]] | None = None,
    callback: Callable[[str, list["Citation"]], None] | None = None,
) -> AsyncGenerator[Event, None]:
    send_text = ""
    send_citations: list[Citation] = []
    current_text = ""
    citation_indices: dict[str, int] = {}

    async for event in event_generator:
        if not event or isinstance(event, Exception):
            yield event
            continue

        elif isinstance(event, RawContentBlockStartEvent):
            send_text = ""
            send_citations = []
            current_text = ""

        elif isinstance(event, RawContentBlockDeltaEvent) and event.delta:
            if isinstance(event.delta, TextDelta):
                delta_text = event.delta.text or ""
                current_text += delta_text

                current_documents = documents_fn() if documents_fn else None
                if current_documents:
                    if not citation_indices:
                        citation_indices = (
                            citation_indices_fn() if citation_indices_fn else {}
                        )

                    result = await asyncio.to_thread(
                        extract_citations_by_original_text,
                        text=current_text,
                        documents=current_documents,
                        citation_indices=citation_indices,
                    )

                    cleaned_text, current_citations, citation_indices = result
                    if not cleaned_text:
                        continue

                    delta_text = cleaned_text[len(send_text) :]
                    delta_citation = current_citations[len(send_citations) :]
                    event.delta = TextDelta.from_citations(delta_text, delta_citation)
                    send_text = cleaned_text
                    send_citations.extend(delta_citation)

                    if not delta_text and not delta_citation:
                        continue

            elif isinstance(event.delta, ThinkingDelta):
                delta_thinking = event.delta.thinking or ""
                current_text += delta_thinking

                current_documents = documents_fn() if documents_fn else None
                if current_documents:
                    if not citation_indices:
                        citation_indices = (
                            citation_indices_fn() if citation_indices_fn else {}
                        )

                    result = await asyncio.to_thread(
                        extract_citations_by_original_text,
                        text=current_text,
                        documents=current_documents,
                        citation_indices=citation_indices,
                    )

                    cleaned_text, current_citations, citation_indices = result
                    if not cleaned_text:
                        continue

                    delta_thinking = cleaned_text[len(send_text) :]
                    delta_citation = current_citations[len(send_citations) :]
                    event.delta = ThinkingDelta.from_citations(
                        delta_thinking, delta_citation
                    )
                    send_text = cleaned_text
                    send_citations.extend(delta_citation)

                    if not delta_thinking and not delta_citation:
                        continue

        yield event

    # Final callback after processing all events
    if callback:
        callback(send_text, send_citations)
