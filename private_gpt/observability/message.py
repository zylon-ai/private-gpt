"""Fold a turn's streamed events into the final assistant message for tracing.

The chat engine streams Anthropic-style events. Backends that render a turn
(session views) need the folded result on the turn span:
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
    from collections.abc import Mapping, Sequence

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


# Inline media up to this size (base64) so trace UIs can render it; larger
# payloads are summarized (type, mime, size) to keep spans small.
_MAX_INLINE_MEDIA = 48_000


def _media(kind: str, encoded: Any, mime: str | None, url: Any) -> dict[str, Any]:
    """Summary of a media block; small images are inlined as a data URL.

    ``encoded`` is the base64 payload llama-index keeps once a block is
    validated. Its size is derived from the length, so nothing is decoded
    or re-encoded per span.
    """
    block: dict[str, Any] = {"type": kind}
    if mime:
        block["media_type"] = mime
    if url:
        block["url"] = str(url)
    if isinstance(encoded, bytes) and encoded:
        padding = encoded[-2:].count(b"=")
        block["size"] = len(encoded) * 3 // 4 - padding
        if kind == "image" and len(encoded) <= _MAX_INLINE_MEDIA:
            block["url"] = f"data:{mime or 'image/png'};base64,{encoded.decode()}"
    return block


def _document(block: Any) -> dict[str, Any]:
    data = block.model_dump(mode="json", exclude_none=True)
    source = data.get("source") or {}
    folded: dict[str, Any] = {"type": "document"}
    for key in ("title", "context"):
        if data.get(key):
            folded[key] = data[key]
    folded["source"] = {
        key: value
        for key, value in source.items()
        if key in ("type", "media_type", "url", "file_id")
    }
    if source.get("type") == "text" and isinstance(source.get("data"), str):
        folded["source"]["text"] = source["data"][:4_000]
    elif isinstance(source.get("data"), str):
        folded["source"]["size"] = len(source["data"]) * 3 // 4
    return folded


def fold_user_message(message: Any) -> dict[str, Any] | None:
    """Anthropic-shaped view of a llama-index user message for tracing.

    Text, images (inline when small), audio and attached documents, the same
    whether the request came as JSON, SSE or through ``/messages/async``.
    """
    content: list[dict[str, Any]] = []
    for block in getattr(message, "blocks", None) or []:
        kind = getattr(block, "block_type", None)
        if kind == "text" and (text := getattr(block, "text", "")).strip():
            content.append({"type": "text", "text": text})
        elif kind == "image":
            content.append(
                _media(
                    "image",
                    block.image,
                    block.image_mimetype,
                    block.url or block.path,
                )
            )
        elif kind == "audio":
            content.append(_media("audio", block.audio, block.format, block.url))
        elif kind == "document":
            content.append(
                {
                    "type": "document",
                    "title": getattr(block, "title", None),
                    "source": {"media_type": getattr(block, "document_mimetype", None)},
                }
            )
    for document in (getattr(message, "additional_kwargs", None) or {}).get(
        "document", []
    ) or []:
        try:
            content.append(_document(document))
        except Exception:
            content.append({"type": "document"})
    if not content:
        return None
    return {"role": "user", "content": content}


def last_user_message(messages: Sequence[Any]) -> dict[str, Any] | None:
    """Folded last user message that has content (tool results have none)."""
    for message in reversed(messages or []):
        role = getattr(message, "role", None)
        if str(getattr(role, "value", role)) != "user":
            continue
        folded = fold_user_message(message)
        if folded is not None:
            return folded
    return None


def message_text(message: Mapping[str, Any] | None) -> str | None:
    """Markdown rendering for span inputs and session bubbles.

    Text as is; images inline (``![image](data:...)``) when small enough so
    backends render them in sessions and span details alike; audio and
    documents as ``[audio: audio/wav]`` / ``[document: report.pdf]`` markers.
    """
    if not message:
        return None
    parts: list[str] = []
    for block in message.get("content", []):
        kind = block.get("type")
        if kind == "text":
            parts.append(block["text"])
        elif kind == "image" and block.get("url"):
            parts.append(f"![{block.get('media_type') or 'image'}]({block['url']})")
        else:
            name = block.get("title") or block.get("media_type") or ""
            size = f", {human_bytes(block['size'])}" if block.get("size") else ""
            parts.append(f"[{kind}{': ' + name if name else ''}{size}]")
    return "\n\n".join(parts) or None


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB"):
        if value < 1024:
            return f"{value:.0f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"
