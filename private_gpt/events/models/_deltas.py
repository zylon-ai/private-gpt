from typing import Any, Literal

from pydantic import Field

from private_gpt.chat.extensions.citation import ZylonCitation
from private_gpt.components.chunk.models import Chunk
from private_gpt.components.engines.citations.types import Citation
from private_gpt.events.models._base import (
    BaseContentBlock,
    ExtendedContentProtocol,
    StandardContentProtocol,
)
from private_gpt.events.models._content_blocks import BasicContentBlockType

# ---------------------------------------------------------------------------
# Standard deltas
# ---------------------------------------------------------------------------


class TextDelta(BaseContentBlock, StandardContentProtocol):
    """Incremental text content."""

    type: Literal["text_delta"] = Field(default="text_delta")
    text: str | None = Field(default=None)
    citations: list[ZylonCitation] | None = Field(default=None)

    @classmethod
    def from_text(cls, text: str) -> "TextDelta":
        return cls(text=text)

    @classmethod
    def from_citations(
        cls, text: str | None, citations: list[Citation] | None
    ) -> "TextDelta":
        return cls(
            text=text,
            citations=(
                [ZylonCitation.from_citation(c) for c in citations]
                if citations
                else None
            ),
        )

    @classmethod
    def from_citation(cls, text: str | None, citation: Citation | None) -> "TextDelta":
        return cls.from_citations(text=text, citations=[citation] if citation else None)


class InputJSONDelta(BaseContentBlock, StandardContentProtocol):
    """Incremental JSON fragment for tool input streaming."""

    type: Literal["input_json_delta"] = Field(default="input_json_delta")
    partial_json: str = Field(default="")
    partial_json_obj: dict[str, Any] = Field(default_factory=dict)


class CitationsDelta(BaseContentBlock, StandardContentProtocol):
    """Citation delta emitted when the model references a source document."""

    type: Literal["citations_delta"] = Field(default="citations_delta")
    citation: ZylonCitation = Field(description="The cited source location")


class ThinkingDelta(BaseContentBlock, StandardContentProtocol):
    """Incremental extended-thinking content."""

    type: Literal["thinking_delta"] = Field(default="thinking_delta")
    thinking: str = Field(description="Incremental reasoning content")
    citations: list[ZylonCitation] | None = Field(default=None)

    @classmethod
    def from_text(cls, thinking: str) -> "ThinkingDelta":
        return cls(thinking=thinking)

    @classmethod
    def from_citations(
        cls, thinking: str, citations: list[Citation] | None
    ) -> "ThinkingDelta":
        return cls(
            thinking=thinking,
            citations=(
                [ZylonCitation.from_citation(c) for c in citations]
                if citations
                else None
            ),
        )


class SignatureDelta(BaseContentBlock, StandardContentProtocol):
    """Incremental extended-thinking signature."""

    type: Literal["signature_delta"] = Field(default="signature_delta")
    signature: str = Field(description="Incremental signature content")


# ---------------------------------------------------------------------------
# Zylon-only deltas
# ---------------------------------------------------------------------------


class SourceDelta(BaseContentBlock, ExtendedContentProtocol):
    """Incremental RAG source attribution."""

    type: Literal["source_delta"] = Field(default="source_delta")
    sources: list[Chunk] = Field(description="Document chunks being streamed")


class TLDRDelta(BaseContentBlock, ExtendedContentProtocol):
    """Incremental TLDR summary block."""

    type: Literal["tldr_delta"] = Field(default="tldr_delta")
    tldr: BasicContentBlockType = Field(description="Summary content block")
    tldr_side: Literal["left", "right"] = Field(default="left")


ContentBlockDeltaType = (
    TextDelta
    | InputJSONDelta
    | CitationsDelta
    | ThinkingDelta
    | SignatureDelta
    | SourceDelta
    | TLDRDelta
)
