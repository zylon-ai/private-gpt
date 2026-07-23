from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from private_gpt.components.text_processing.models import (
    Action,
    ProbeResult,
    ProbeStatus,
    ProcessingContext,
)


class StreamRule(Protocol):
    name: str
    priority: int

    def probe(
        self, text: str, position: int, context: ProcessingContext
    ) -> ProbeResult: ...


ResolveReferences = Callable[[list[str], ProcessingContext], list[Any]]
RenderReferences = Callable[[list[Any], ProcessingContext], tuple[str, tuple[Any, ...]]]


@dataclass
class DelimitedReferenceRule:
    start_token: str
    end_token: str
    separator: str
    resolve: ResolveReferences
    render: RenderReferences
    name: str = "delimited_reference"
    priority: int = 100

    def probe(
        self, text: str, position: int, context: ProcessingContext
    ) -> ProbeResult:
        if not text.startswith(self.start_token, position):
            return ProbeResult.no_match()

        # Skip consecutive start tokens (e.g., [[[ for [[[XXXX]]])
        content_start = position + len(self.start_token)
        while content_start < len(text) and text.startswith(
            self.start_token, content_start
        ):
            content_start += len(self.start_token)

        end = text.find(self.end_token, content_start)
        if end == -1:
            return ProbeResult.need_more()

        # Skip consecutive end tokens (e.g., ]]] for [[[XXXX]]])
        end_offset = end + len(self.end_token)
        while end_offset < len(text) and text.startswith(self.end_token, end_offset):
            end_offset += len(self.end_token)

        consumed = end_offset - position
        identifiers = [
            identifier.strip()
            for identifier in text[content_start:end].split(self.separator)
        ]
        references = self.resolve(identifiers, context)
        if not references:
            return ProbeResult(
                status=ProbeStatus.MATCH,
                consumed=consumed,
                action=Action.PASS,
            )

        replacement, metadata = self.render(references, context)
        return ProbeResult(
            status=ProbeStatus.MATCH,
            consumed=consumed,
            action=Action.REPLACE,
            replacement=replacement,
            metadata=metadata,
        )


@dataclass
class BacktickUnwrapRule:
    inner: StreamRule
    name: str = "backtick_unwrap"
    priority: int = 200
    code_state_key: str = "backtick_code_delimiter"
    wrapper_state_key: str = "backtick_wrapper_delimiter"

    def probe(
        self, text: str, position: int, context: ProcessingContext
    ) -> ProbeResult:
        if text[position] != "`":
            return ProbeResult.no_match()

        delimiter_end = position + 1
        while delimiter_end < len(text) and text[delimiter_end] == "`":
            delimiter_end += 1
        delimiter = text[position:delimiter_end]

        if context.state.get(self.wrapper_state_key) == delimiter:
            return ProbeResult(
                status=ProbeStatus.MATCH,
                consumed=len(delimiter),
                action=Action.DROP,
                state_deletes=(self.wrapper_state_key,),
            )

        if context.state.get(self.code_state_key) == delimiter:
            return ProbeResult(
                status=ProbeStatus.MATCH,
                consumed=len(delimiter),
                action=Action.PASS,
                state_deletes=(self.code_state_key,),
            )

        if delimiter_end == len(text):
            if not context.final:
                return ProbeResult.need_more()
            return ProbeResult(
                status=ProbeStatus.MATCH,
                consumed=len(delimiter),
                action=Action.PASS,
            )

        inner_match = self.inner.probe(text, delimiter_end, context)
        if inner_match.status == ProbeStatus.NEED_MORE:
            return inner_match
        if (
            inner_match.status == ProbeStatus.MATCH
            and inner_match.action == Action.REPLACE
        ):
            consumed = len(delimiter) + inner_match.consumed
            updates = dict(inner_match.state_updates)
            deletes = list(inner_match.state_deletes)
            if text.startswith(delimiter, position + consumed):
                consumed += len(delimiter)
            else:
                updates[self.wrapper_state_key] = delimiter
            return ProbeResult(
                status=ProbeStatus.MATCH,
                consumed=consumed,
                action=Action.UNWRAP,
                replacement=inner_match.replacement,
                metadata=inner_match.metadata,
                state_updates=updates,
                state_deletes=tuple(deletes),
            )

        return ProbeResult(
            status=ProbeStatus.MATCH,
            consumed=len(delimiter),
            action=Action.PASS,
            state_updates={self.code_state_key: delimiter},
        )


@dataclass
class LooseReferenceCleanupRule:
    start_token: str
    end_token: str
    identifier_length: int
    identifiers: tuple[str, ...]
    name: str = "loose_reference_cleanup"
    priority: int = 50

    def __post_init__(self) -> None:
        self._pattern = re.compile(
            rf"{re.escape(self.start_token)}?[A-Z0-9]"
            rf"{{{self.identifier_length}}}{re.escape(self.end_token)}?"
        )

    def probe(
        self, text: str, position: int, context: ProcessingContext
    ) -> ProbeResult:
        match = self._pattern.match(text, position)
        if match is None:
            return ProbeResult.no_match()
        word = match.group(0)
        if word.startswith(self.start_token) and word.endswith(self.end_token):
            return ProbeResult.no_match()
        identifier = next(
            (identifier for identifier in self.identifiers if identifier in word),
            None,
        )
        if identifier is None:
            return ProbeResult.no_match()
        replacement = word.replace(identifier, "", 1).strip()
        return ProbeResult(
            status=ProbeStatus.MATCH,
            consumed=len(word),
            action=Action.REPLACE,
            replacement=replacement,
        )
