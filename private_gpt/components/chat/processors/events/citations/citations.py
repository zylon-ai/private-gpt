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
    RawContentBlockStopEvent,
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
    last_delta_type: str | None = None
    last_block_id: str | None = None

    async for event in event_generator:
        if not event or isinstance(event, Exception):
            yield event
            continue

        elif isinstance(event, RawContentBlockStartEvent):
            send_text = ""
            send_citations = []
            current_text = ""
            last_delta_type = None
            last_block_id = event.block_id

        elif isinstance(event, RawContentBlockDeltaEvent) and event.delta:
            if isinstance(event.delta, TextDelta):
                last_delta_type = "text"
                last_block_id = event.block_id
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
                        is_final=False,
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
                last_delta_type = "thinking"
                last_block_id = event.block_id
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
                        is_final=False,
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

        elif isinstance(event, RawContentBlockStopEvent):
            current_documents = documents_fn() if documents_fn else None
            if current_documents and current_text:
                if not citation_indices:
                    citation_indices = (
                        citation_indices_fn() if citation_indices_fn else {}
                    )

                result = await asyncio.to_thread(
                    extract_citations_by_original_text,
                    text=current_text,
                    documents=current_documents,
                    citation_indices=citation_indices,
                    is_final=True,
                )

                cleaned_text, current_citations, citation_indices = result
                delta_text = cleaned_text[len(send_text) :]
                delta_citation = current_citations[len(send_citations) :]

                if delta_text or delta_citation:
                    if last_delta_type == "text":
                        delta = TextDelta.from_citations(delta_text, delta_citation)
                    elif last_delta_type == "thinking":
                        delta = ThinkingDelta.from_citations(delta_text, delta_citation)
                    else:
                        delta = TextDelta.from_citations(delta_text, delta_citation)

                    yield RawContentBlockDeltaEvent(block_id=event.block_id, delta=delta)
                    
                    send_text = cleaned_text
                    send_citations.extend(delta_citation)

        yield event

    current_documents = documents_fn() if documents_fn else None
    if current_documents and current_text:
        if not citation_indices:
            citation_indices = (
                citation_indices_fn() if citation_indices_fn else {}
            )
        result = await asyncio.to_thread(
            extract_citations_by_original_text,
            text=current_text,
            documents=current_documents,
            citation_indices=citation_indices,
            is_final=True,
        )
        cleaned_text, current_citations, citation_indices = result
        send_text = cleaned_text
        send_citations = current_citations

    # Final callback after processing all events
    if callback:
        callback(send_text, send_citations)
