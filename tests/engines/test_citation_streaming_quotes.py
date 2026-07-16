from private_gpt.components.engines.citations.types import Document
from private_gpt.components.engines.citations.utils import (
    extract_citations_by_original_text,
)


def test_streaming_citation_examples_do_not_rewrite_previous_text() -> None:
    text = (
        "Format: `[XXXX]`. Correct: `[LLFB]`. "
        "Invalid: `[[LLFB]]`, `(LLFB)`. "
        "Consolidate: `[LLFB], [CPCY]`. End."
    )
    documents = [
        Document(
            type="document",
            id_=f"source-{citation_id}",
            shorter_id=citation_id,
            document_id="artifact",
            text=citation_id,
        )
        for citation_id in ("LLFB", "CPCY")
    ]
    current_text = ""
    sent_text = ""
    emitted_text = ""
    citation_indices: dict[str, int] = {}

    for character in text:
        current_text += character
        cleaned_text, _, citation_indices = extract_citations_by_original_text(
            current_text,
            documents,
            citation_indices=citation_indices,
        )
        assert cleaned_text.startswith(sent_text)
        emitted_text += cleaned_text[len(sent_text) :]
        sent_text = cleaned_text

    assert "Format: `[XXXX]`." in emitted_text
    assert "Invalid: `[[LLFB]]`, `(LLFB)`." in emitted_text
    assert "Consolidate: <citation id='LLFB'" in emitted_text
    assert ", <citation id='CPCY'" in emitted_text
    assert "</citation>`. End." not in emitted_text
