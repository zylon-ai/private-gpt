from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from private_gpt.components.text_processing.models import (
    Action,
    ProbeStatus,
    ProcessDelta,
    ProcessingContext,
    ProcessResult,
)

if TYPE_CHECKING:
    from private_gpt.components.text_processing.rules import StreamRule


class IncrementalTextProcessor:
    def __init__(
        self,
        rules: list[StreamRule],
        initial_state: dict[str, object] | None = None,
    ) -> None:
        self._rules = sorted(rules, key=lambda rule: rule.priority, reverse=True)
        self._initial_state = deepcopy(initial_state or {})
        self._source = ""
        self._emitted = ""
        self._metadata_count = 0
        self.context = ProcessingContext(state=deepcopy(self._initial_state))

    def process(
        self,
        text: str,
        *,
        final: bool = False,
        context: ProcessingContext | None = None,
    ) -> ProcessResult:
        active_context = context or ProcessingContext()
        active_context.final = final
        output: list[str] = []
        metadata: list[object] = []
        cursor = 0

        while cursor < len(text):
            probe = None
            for rule in self._rules:
                candidate = rule.probe(text, cursor, active_context)
                if candidate.status != ProbeStatus.NO_MATCH:
                    probe = candidate
                    break

            if probe is None:
                output.append(text[cursor])
                cursor += 1
                continue

            if probe.status == ProbeStatus.NEED_MORE:
                break
            if probe.consumed <= 0:
                raise ValueError("A matching stream rule must consume source text")

            source = text[cursor : cursor + probe.consumed]
            if probe.action == Action.PASS:
                output.append(
                    probe.replacement if probe.replacement is not None else source
                )
            elif probe.action in (Action.REPLACE, Action.UNWRAP):
                output.append(probe.replacement or "")
            elif probe.action == Action.DROP:
                pass
            else:
                raise ValueError(f"Unsupported matching action: {probe.action}")

            for key in probe.state_deletes:
                active_context.state.pop(key, None)
            active_context.state.update(probe.state_updates)
            metadata.extend(probe.metadata)
            cursor += probe.consumed

        return ProcessResult(
            text="".join(output),
            metadata=tuple(metadata),
            pending=text[cursor:],
            consumed=cursor,
        )

    def feed(self, chunk: str) -> ProcessDelta:
        self._source += chunk
        self.context = ProcessingContext(state=deepcopy(self._initial_state))
        result = self.process(self._source, context=self.context)
        if not result.text.startswith(self._emitted):
            raise ValueError("A stream rule rewrote an already-emitted prefix")
        delta = ProcessDelta(
            text=result.text[len(self._emitted) :],
            metadata=result.metadata[self._metadata_count :],
            pending=result.pending,
        )
        self._emitted = result.text
        self._metadata_count = len(result.metadata)
        return delta

    def finalize(self) -> ProcessDelta:
        self.context = ProcessingContext(state=deepcopy(self._initial_state))
        result = self.process(self._source, final=True, context=self.context)
        if not result.text.startswith(self._emitted):
            raise ValueError("Finalization rewrote an already-emitted prefix")
        delta = ProcessDelta(
            text=result.text[len(self._emitted) :],
            metadata=result.metadata[self._metadata_count :],
            pending=result.pending,
        )
        self._emitted = result.text
        self._metadata_count = len(result.metadata)
        return delta
