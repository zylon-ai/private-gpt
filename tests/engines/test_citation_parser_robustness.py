import pytest

from private_gpt.components.engines.citations.types import Document
from private_gpt.components.engines.citations.utils import (
    extract_citations_by_original_text,
    format_cite,
)


def create_document(citation_id: str) -> Document:
    return Document(
        type="document",
        id_=f"source-{citation_id}",
        shorter_id=citation_id,
        document_id=f"artifact-{citation_id}",
        text=f"Content for {citation_id}",
    )


def test_repeated_citation_keeps_index_but_emits_each_occurrence() -> None:
    document = create_document("AB12")

    formatted, citations, indices = extract_citations_by_original_text(
        "First [AB12], repeated [AB12].",
        [document],
    )

    assert formatted == (
        f"First {format_cite(0, document, 0)}, repeated {format_cite(1, document, 0)}."
    )
    assert [citation.value["index"] for citation in citations] == ["0", "0"]
    assert indices == {document.id_: 0}


def test_mixed_known_and_unknown_consolidated_citation_keeps_known_only() -> None:
    document = create_document("AB12")

    formatted, citations, _ = extract_citations_by_original_text(
        "Claim [UNKNOWN, AB12, MISSING].",
        [document],
    )

    assert formatted == f"Claim {format_cite(0, document, 0)}."
    assert len(citations) == 1


def test_citation_lookup_is_case_insensitive() -> None:
    document = create_document("AB12")

    formatted, citations, _ = extract_citations_by_original_text(
        "Claim [ab12].",
        [document],
    )

    assert formatted == f"Claim {format_cite(0, document, 0)}."
    assert len(citations) == 1


def test_unicode_citation_brackets_are_normalized() -> None:
    document = create_document("AB12")

    formatted, citations, _ = extract_citations_by_original_text(
        "Claim 【AB12】.",
        [document],
    )

    assert formatted == f"Claim {format_cite(0, document, 0)}."
    assert len(citations) == 1


@pytest.mark.parametrize("delimiter", ["`", "``", "```"])
def test_backtick_wrapped_citation_removes_matching_delimiter(delimiter: str) -> None:
    document = create_document("AB12")

    formatted, citations, _ = extract_citations_by_original_text(
        f"Claim {delimiter}[AB12]{delimiter}.",
        [document],
    )

    assert formatted == f"Claim {format_cite(0, document, 0)}."
    assert len(citations) == 1


@pytest.mark.parametrize(
    "garbage",
    [
        "[]",
        "[ ]",
        "[[AB12]]",
        "(AB12)",
        "[AB-12]",
        "[TOO-LONG]",
        "[UNKNOWN]",
        "prefix AB12 suffix",
    ],
)
def test_non_citation_garbage_is_preserved(garbage: str) -> None:
    document = create_document("AB12")

    formatted, citations, _ = extract_citations_by_original_text(
        garbage,
        [document],
    )

    assert formatted == garbage
    assert citations == []


def test_incomplete_citation_is_withheld_with_no_false_citation() -> None:
    document = create_document("AB12")

    formatted, citations, _ = extract_citations_by_original_text(
        "Safe prefix [AB1",
        [document],
    )

    assert formatted == "Safe prefix "
    assert citations == []


@pytest.mark.xfail(
    strict=True,
    reason="Model output can currently collide with the internal citation placeholder.",
)
def test_placeholder_like_model_output_does_not_capture_real_citation() -> None:
    document = create_document("AB12")
    model_text = "Literal \ue000citationn0\ue001 then [AB12]."

    formatted, citations, _ = extract_citations_by_original_text(
        model_text,
        [document],
    )

    assert formatted == (
        "Literal \ue000citationn0\ue001 then "
        f"{format_cite(0, document, 0)}."
    )
    assert len(citations) == 1


def test_existing_indices_continue_without_renumbering() -> None:
    first = create_document("AB12")
    second = create_document("CD34")

    formatted, citations, indices = extract_citations_by_original_text(
        "Existing [AB12], new [CD34].",
        [first, second],
        citation_indices={first.id_: 7},
    )

    assert formatted == (
        f"Existing {format_cite(0, first, 7)}, new {format_cite(1, second, 8)}."
    )
    assert [citation.value["index"] for citation in citations] == ["7", "8"]
    assert indices == {first.id_: 7, second.id_: 8}
