import random

import pytest

from private_gpt.components.engines.citations.types import Document
from private_gpt.components.engines.citations.utils import (
    extract_citations_by_original_text,
)


TEXT = (
    "Format: `[XXXX]`. Correct: `[AB12]`. "
    "Invalid: `[[AB12]]`, `(AB12)`. "
    "Consolidate: `[AB12], [CD34]`. Final [EF56]."
)


def create_documents() -> list[Document]:
    return [
        Document(
            type="document",
            id_=f"source-{citation_id}",
            shorter_id=citation_id,
            document_id=f"artifact-{citation_id}",
            text=citation_id,
        )
        for citation_id in ("AB12", "CD34", "EF56")
    ]


def stream_chunks(chunks: list[str]) -> tuple[str, int]:
    documents = create_documents()
    current_text = ""
    sent_text = ""
    emitted_text = ""
    citation_indices: dict[str, int] = {}
    citation_count = 0

    for chunk in chunks:
        current_text += chunk
        cleaned_text, citations, citation_indices = extract_citations_by_original_text(
            current_text,
            documents,
            citation_indices=citation_indices,
        )
        assert cleaned_text.startswith(sent_text)
        emitted_text += cleaned_text[len(sent_text) :]
        sent_text = cleaned_text
        citation_count = len(citations)

    final_text, citations, _ = extract_citations_by_original_text(
        current_text,
        documents,
        citation_indices=citation_indices,
        is_final=True,
    )
    assert final_text.startswith(sent_text)
    emitted_text += final_text[len(sent_text) :]
    return emitted_text, len(citations) if citations else citation_count


def expected_result() -> tuple[str, int]:
    formatted, citations, _ = extract_citations_by_original_text(
        TEXT,
        create_documents(),
        is_final=True,
    )
    return formatted, len(citations)


def test_every_two_chunk_split_matches_one_shot_result() -> None:
    expected_text, expected_citations = expected_result()

    for split_at in range(len(TEXT) + 1):
        streamed_text, citation_count = stream_chunks(
            [TEXT[:split_at], TEXT[split_at:]]
        )
        assert streamed_text == expected_text, f"split_at={split_at}"
        assert citation_count == expected_citations, f"split_at={split_at}"


def test_character_by_character_stream_matches_one_shot_result() -> None:
    assert stream_chunks(list(TEXT)) == expected_result()


@pytest.mark.parametrize("seed", range(20))
def test_random_chunking_matches_one_shot_result(seed: int) -> None:
    random_generator = random.Random(seed)
    chunks = []
    position = 0
    while position < len(TEXT):
        chunk_size = random_generator.randint(1, 12)
        chunks.append(TEXT[position : position + chunk_size])
        position += chunk_size

    assert stream_chunks(chunks) == expected_result()


@pytest.mark.parametrize(
    "partial",
    [
        "Prefix [",
        "Prefix [A",
        "Prefix [AB",
        "Prefix [AB1",
        "Prefix `[AB12",
        "Prefix ``[AB12",
        "Prefix ```[AB12",
    ],
)
def test_partial_citation_never_leaks_during_stream(partial: str) -> None:
    formatted, citations, _ = extract_citations_by_original_text(
        partial,
        create_documents(),
    )

    assert formatted == "Prefix "
    assert citations == []
