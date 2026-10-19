"""Fold a turn's streamed events into the final assistant message for tracing.

The chat engine streams Anthropic-style events. Backends that render a turn
(Phoenix sessions, Langfuse, ...) need the folded result on the turn span:
the text the user saw, reasoning, tool calls with their results, stop reason
and usage. ``RecordingEventChannel`` (engines/chat/event_channel.py) feeds a
``MessageFolder`` as events pass, without changing what is delivered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from private_gpt.events.models import (
    InputJSONDelta,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawMessageDeltaEvent,
    TextDelta,
    ThinkingDelta,
)

if TYPE_CHECKING:
    from private_gpt.events.models import Event

# Fields that only matter on the wire (timestamps, cache hints, signatures).
_NOISE = {
    "start_timestamp",
    "stop_timestamp",
    "cache_control",
    "signature",
    "encrypted_content",
}


def _dump(block: Any) -> dict[str, Any]:
    data = block.model_dump(mode="json", exclude_none=True)
    return {k: v for k, v in data.items() if k not in _NOISE}


class MessageFolder:
    """Accumulates one assistant message from its streamed events."""

    def __init__(self) -> None:
        self._blocks: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self.stop_reason: str | None = None
        self.usage: dict[str, Any] | None = None

    def add(self, event: Event) -> None:
        if isinstance(event, RawContentBlockStartEvent) and event.content_block:
            if event.block_id not in self._blocks:
                self._order.append(event.block_id)
            self._blocks[event.block_id] = _dump(event.content_block)
        elif isinstance(event, RawContentBlockDeltaEvent) and event.delta:
            block = self._blocks.get(event.block_id)
            if block is None:
                return
            delta = event.delta
            if isinstance(delta, TextDelta):
                block["text"] = (block.get("text") or "") + (delta.text or "")
            elif isinstance(delta, ThinkingDelta):
                block["thinking"] = (block.get("thinking") or "") + delta.thinking
            elif isinstance(delta, InputJSONDelta) and delta.partial_json_obj:
                block["input"] = delta.partial_json_obj
        elif isinstance(event, RawMessageDeltaEvent):
            if event.delta and event.delta.stop_reason:
                self.stop_reason = event.delta.stop_reason
            if event.usage:
                self.usage = event.usage.model_dump(mode="json", exclude_none=True)

    @property
    def content(self) -> list[dict[str, Any]]:
        return [self._blocks[block_id] for block_id in self._order]

    @property
    def text(self) -> str:
        """The answer text as the user read it (text blocks only)."""
        return "".join(
            block.get("text") or ""
            for block in self.content
            if block.get("type") == "text"
        )

    @property
    def reasoning(self) -> str:
        return "".join(
            block.get("thinking") or ""
            for block in self.content
            if block.get("type") == "thinking"
        )

    def message(self) -> dict[str, Any]:
        """Anthropic-shaped message: role, content blocks, stop reason, usage."""
        message: dict[str, Any] = {
            "role": "assistant",
            "content": self.content,
            "stop_reason": self.stop_reason,
        }
        if self.usage:
            message["usage"] = self.usage
        return message
