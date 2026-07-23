from __future__ import annotations

from collections.abc import Callable

from private_gpt.components.engines.citations.types import Document
from private_gpt.components.text_processing import (
    BacktickUnwrapRule,
    DelimitedReferenceRule,
    IncrementalTextProcessor,
    ProcessingContext,
)

CitationFormatter = Callable[[int, Document, int], str]


class CitationTextParser:
    def __init__(
        self,
        documents: list[Document],
        formatter: CitationFormatter,
        *,
        start_token: str,
        end_token: str,
        separator: str,
        identifier_length: int,
        citation_indices: dict[str, int] | None = None,
    ) -> None:
        self._documents_by_id = {
            document.id.lower(): document for document in documents
        }
        self._start_token = start_token
        self._end_token = end_token
        self._formatter = formatter
        self._initial_indices = dict(citation_indices or {})
        self._context = ProcessingContext(
            state={
                "citation_indices": dict(self._initial_indices),
                "citation_next_index": max(self._initial_indices.values(), default=-1)
                + 1,
                "citation_occurrence": 0,
            }
        )
        reference_rule = DelimitedReferenceRule(
            start_token=start_token,
            end_token=end_token,
            separator=separator,
            resolve=self._resolve,
            render=self._render,
        )
        self._processor = IncrementalTextProcessor(
            [
                BacktickUnwrapRule(reference_rule),
                reference_rule,
            ]
        )

    def parse(self, text: str, *, final: bool = False) -> tuple[str, dict[str, int]]:
        normalized = text.replace("【", self._start_token).replace(
            "】", self._end_token
        )
        result = self._processor.process(
            normalized,
            final=final,
            context=self._context,
        )
        return result.text, dict(self._context.state["citation_indices"])

    def _resolve(
        self, identifiers: list[str], context: ProcessingContext
    ) -> list[Document]:
        return [
            self._documents_by_id[identifier.lower()]
            for identifier in identifiers
            if identifier.lower() in self._documents_by_id
        ]

    def _render(
        self, documents: list[Document], context: ProcessingContext
    ) -> tuple[str, tuple[Document, ...]]:
        indices: dict[str, int] = context.state["citation_indices"]
        next_index: int = context.state["citation_next_index"]
        occurrence: int = context.state["citation_occurrence"]
        rendered = []

        for document in documents:
            if document.id_ in indices:
                index = indices[document.id_]
            else:
                index = next_index
                indices[document.id_] = index
                next_index += 1
            rendered.append(self._formatter(occurrence, document, index))
            occurrence += 1

        context.state["citation_next_index"] = next_index
        context.state["citation_occurrence"] = occurrence
        return ",".join(rendered), tuple(documents)
