from __future__ import annotations

import base64
import re
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from llama_index.core.base.llms.types import AudioBlock as LIAudioBlock
from llama_index.core.base.llms.types import ImageBlock as LIImageBlock
from llama_index.core.base.llms.types import TextBlock as LITextBlock
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    model_serializer,
    model_validator,
)

from private_gpt.chat.extensions.citation import ZylonCitation  # noqa: TC001
from private_gpt.components.chunk.models import Chunk, SourceType  # noqa: TC001
from private_gpt.events.models._base import (
    BaseContentBlock,
    CacheableContentBlock,
    ExtendedContentProtocol,
    StandardContentProtocol,
)
from private_gpt.events.models._callers import ToolCaller  # noqa: TC001

if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore
    from PIL.Image import Image
    from pydantic_core.core_schema import SerializerFunctionWrapHandler


class TextBlock(CacheableContentBlock, StandardContentProtocol):
    """Plain-text content block."""

    type: Literal["text"] = Field(default="text")
    text: str = Field(default="", description="Text payload for this block.")
    citations: list[ZylonCitation] | None = Field(default=[])

    def to_llama_index(self) -> LITextBlock:
        return LITextBlock(text=self.text)

    def __str__(self) -> str:
        return self.text or ""

    @model_serializer(mode="wrap")
    def custom_model_dump(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        data: dict[str, Any] = super().custom_model_dump(handler)
        if not data.get("citations"):
            data.pop("citations", None)
        return data


class URLSource(BaseModel):
    """Shared URL source payload (accepts url/uri input)."""

    type: Literal["url"]
    url: str = Field(
        description="Publicly reachable URL",
        validation_alias=AliasChoices("url", "uri"),
        serialization_alias="url",
    )
    model_config = ConfigDict(extra="allow")


class CitationsConfig(BaseModel):
    enabled: bool | None = Field(default=None)

    model_config = ConfigDict(extra="allow")


class CitationCharLocation(BaseModel):
    type: Literal["char_location"] = Field(default="char_location")
    cited_text: str
    document_index: int
    document_title: str | None
    start_char_index: int
    end_char_index: int

    model_config = ConfigDict(extra="allow")


class CitationPageLocation(BaseModel):
    type: Literal["page_location"] = Field(default="page_location")
    cited_text: str
    document_index: int
    document_title: str | None
    start_page_number: int
    end_page_number: int

    model_config = ConfigDict(extra="allow")


class CitationContentBlockLocation(BaseModel):
    type: Literal["content_block_location"] = Field(default="content_block_location")
    cited_text: str
    document_index: int
    document_title: str | None
    start_block_index: int
    end_block_index: int

    model_config = ConfigDict(extra="allow")


class CitationSearchResultLocation(BaseModel):
    type: Literal["search_result_location"] = Field(default="search_result_location")
    cited_text: str
    search_result_index: int
    source: str
    title: str | None
    start_block_index: int
    end_block_index: int

    model_config = ConfigDict(extra="forbid")


class CitationWebSearchResultLocation(BaseModel):
    type: Literal["web_search_result_location"] = Field(
        default="web_search_result_location"
    )
    cited_text: str
    encrypted_index: str
    title: str | None
    url: str

    model_config = ConfigDict(extra="forbid")


TextCitation = Annotated[
    CitationCharLocation
    | CitationPageLocation
    | CitationContentBlockLocation
    | CitationSearchResultLocation
    | CitationWebSearchResultLocation,
    Field(discriminator="type"),
]

_URI_WITH_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def _upgrade_legacy_source_payload(
    values: Any, *, normalize_image_media_type: bool = False
) -> Any:
    if not isinstance(values, dict) or "source" in values:
        return values

    upgraded = dict(values)
    url = upgraded.pop("url", None) or upgraded.pop("uri", None)
    if isinstance(url, str):
        upgraded["source"] = {"type": "url", "url": url}
        upgraded.pop("data", None)
        upgraded.pop("mime_type", None)
        return upgraded

    data = upgraded.get("data")
    mime_type = upgraded.get("mime_type")
    if isinstance(data, str) and isinstance(mime_type, str):
        if _URI_WITH_SCHEME_RE.match(data):
            upgraded["source"] = {"type": "url", "url": data}
        else:
            media_type = (
                _normalize_image_media_type(mime_type)
                if normalize_image_media_type
                else mime_type
            )
            upgraded["source"] = {
                "type": "base64",
                "data": data,
                "media_type": media_type,
            }
        upgraded.pop("data", None)
        upgraded.pop("mime_type", None)
    return upgraded


def _upgrade_legacy_binary_source_payload(values: Any) -> Any:
    if not isinstance(values, dict):
        return values

    upgraded = dict(values)
    source = upgraded.get("source")
    if isinstance(source, dict) and source.get("type") == "url":
        url = source.get("url") or source.get("uri")
        if isinstance(url, str):
            upgraded["source"] = {"type": "url", "url": url}
        return upgraded

    if "source" in upgraded:
        return values

    url = upgraded.pop("url", None) or upgraded.pop("uri", None)
    if isinstance(url, str):
        upgraded["source"] = {"type": "url", "url": url}
        upgraded.pop("data", None)
        upgraded.pop("mime_type", None)
        return upgraded

    data = upgraded.get("data")
    mime_type = upgraded.get("mime_type")
    if isinstance(data, str) and isinstance(mime_type, str):
        if _URI_WITH_SCHEME_RE.match(data):
            upgraded["source"] = {"type": "url", "url": data}
        else:
            upgraded["source"] = {
                "type": "base64",
                "data": data,
                "media_type": mime_type,
            }
        upgraded.pop("data", None)
        upgraded.pop("mime_type", None)
    return upgraded


ImageMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


def _normalize_image_media_type(mime_type: str) -> ImageMediaType:
    allowed = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    return cast(ImageMediaType, mime_type) if mime_type in allowed else "image/png"


class Base64ImageSource(BaseModel):
    """Anthropic base64 image source payload."""

    type: Literal["base64"]
    data: str = Field(
        description="Base64-encoded image bytes",
        json_schema_extra={"format": "byte"},
    )
    media_type: ImageMediaType

    model_config = ConfigDict(extra="allow")


ImageSource = Annotated[Base64ImageSource | URLSource, Field(discriminator="type")]


class ImageBlock(CacheableContentBlock, StandardContentProtocol):
    """Anthropic-compatible image content block."""

    type: Literal["image"] = Field(default="image")
    source: ImageSource = Field(description="Anthropic image source payload")

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_payload(cls, values: Any) -> Any:
        return _upgrade_legacy_source_payload(values, normalize_image_media_type=True)

    @classmethod
    def from_image(cls, image: Image) -> ImageBlock:
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return cls(
            source=Base64ImageSource(
                type="base64",
                data=base64.b64encode(buf.getvalue()).decode(),
                media_type="image/png",
            )
        )

    @classmethod
    def from_base64(cls, data: str, mime_type: str) -> ImageBlock:
        return cls(
            source=Base64ImageSource(
                type="base64",
                data=data,
                media_type=_normalize_image_media_type(mime_type),
            )
        )

    @classmethod
    def from_url(cls, url: str) -> ImageBlock:
        return cls(source=URLSource(type="url", url=url))

    def to_llama_index(self) -> LIImageBlock:
        if isinstance(self.source, URLSource):
            return LIImageBlock(url=self.source.url)
        return LIImageBlock(
            image=base64.b64decode(self.source.data),
            image_mimetype=self.source.media_type,
        )


class Base64AudioSource(BaseModel):
    """Anthropic-style base64 audio source payload."""

    type: Literal["base64"]
    data: str = Field(description="Base64-encoded audio bytes")
    media_type: str = Field(description="Audio MIME type, e.g. 'audio/mpeg'")

    model_config = ConfigDict(extra="allow")


AudioSource = Annotated[Base64AudioSource | URLSource, Field(discriminator="type")]


class AudioBlock(CacheableContentBlock, StandardContentProtocol):
    """Anthropic-compatible audio content block."""

    type: Literal["audio"] = Field(default="audio")
    source: AudioSource = Field(description="Audio source payload")

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_payload(cls, values: Any) -> Any:
        return _upgrade_legacy_source_payload(values)

    @classmethod
    def from_audio(cls, audio: bytes, mime_type: str) -> AudioBlock:
        return cls(
            source=Base64AudioSource(
                type="base64",
                data=base64.b64encode(audio).decode(),
                media_type=mime_type,
            )
        )

    @classmethod
    def from_base64(cls, data: str, mime_type: str) -> AudioBlock:
        return cls(
            source=Base64AudioSource(type="base64", data=data, media_type=mime_type)
        )

    @classmethod
    def from_url(cls, url: str) -> AudioBlock:
        return cls(source=URLSource(type="url", url=url))

    def to_llama_index(self) -> LIAudioBlock:
        if isinstance(self.source, URLSource):
            raise ValueError("URL-backed audio cannot be converted to LlamaIndex bytes")
        return LIAudioBlock(
            audio=base64.b64decode(self.source.data),
            format=self.source.media_type,
        )


class ThinkingBlock(CacheableContentBlock, StandardContentProtocol):
    """Extended thinking block containing the model's reasoning process."""

    type: Literal["thinking"] = Field(default="thinking")
    thinking: str = Field(default="", description="Thinking payload")
    signature: str = Field(
        description="Anthropic reasoning signature required for extended thinking compatibility",
    )
    citations: list[ZylonCitation] | None = Field(default=[])


class RedactedThinkingBlock(CacheableContentBlock, StandardContentProtocol):
    """Redacted thinking block returned when thinking content is encrypted."""

    type: Literal["redacted_thinking"] = Field(default="redacted_thinking")
    data: str = Field(description="Encrypted thinking payload")


class ToolUseBlock(CacheableContentBlock, StandardContentProtocol):
    """Represents a model-initiated tool call."""

    type: Literal["tool_use"] = Field(default="tool_use")
    id: str = Field(
        description="Unique identifier for this tool use",
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    name: str = Field(
        description="Name of the tool being called", min_length=1, max_length=200
    )
    input: dict[str, Any] = Field(
        description="Input payload for the tool call",
        title="ToolUseInput",
    )
    caller: ToolCaller | None = Field(default=None)


class ServerToolUseBlock(CacheableContentBlock, StandardContentProtocol):
    """Represents a server-side (built-in) tool call initiated by the model."""

    type: Literal["server_tool_use"] = Field(default="server_tool_use")
    id: str = Field(
        description="Unique identifier for this server tool use",
        pattern=r"^srvtoolu_[a-zA-Z0-9_]+$",
    )
    name: Literal[
        "web_search",
        "web_fetch",
        "code_execution",
        "bash_code_execution",
        "text_editor_code_execution",
        "tool_search_tool_regex",
        "tool_search_tool_bm25",
    ] = Field(description="Name of the server tool being called")
    input: dict[str, Any] = Field(
        description="Input payload for the server tool call",
        title="ServerToolUseInput",
    )
    caller: ToolCaller | None = Field(default=None)


class ContainerUploadBlock(CacheableContentBlock, StandardContentProtocol):
    """References a file that was uploaded to an Anthropic container."""

    type: Literal["container_upload"] = Field(default="container_upload")
    file_id: str = Field(description="Container file identifier")


class DocumentBlock(CacheableContentBlock, StandardContentProtocol):
    """Anthropic document block (used for document-grounded generation)."""

    class Base64PDFSource(BaseModel):
        type: Literal["base64"]
        data: str = Field(json_schema_extra={"format": "byte"})
        media_type: Literal["application/pdf"]

        model_config = ConfigDict(extra="allow")

    class PlainTextSource(BaseModel):
        type: Literal["text"]
        data: str
        media_type: Literal["text/plain"]

        model_config = ConfigDict(extra="allow")

    class ContentSource(BaseModel):
        type: Literal["content"]
        content: (
            str | list[Annotated[TextBlock | ImageBlock, Field(discriminator="type")]]
        )

        model_config = ConfigDict(extra="allow")

    class URLPDFSource(BaseModel):
        type: Literal["url"]
        url: str

        model_config = ConfigDict(extra="allow")

    type: Literal["document"] = Field(default="document")
    source: Annotated[
        Base64PDFSource | PlainTextSource | ContentSource | URLPDFSource,
        Field(discriminator="type"),
    ] = Field(description="Document source payload")
    title: str | None = Field(default=None)
    context: str | None = Field(default=None, min_length=1)
    citations: list[ZylonCitation] | None = Field(default=None)


class SearchResultBlock(CacheableContentBlock, StandardContentProtocol):
    """Anthropic search result block."""

    type: Literal["search_result"] = Field(default="search_result")
    source: str = Field(description="Search result source payload")
    title: str = Field(description="Search result title")
    content: list[TextBlock] = Field(description="Search result content")
    citations: list[ZylonCitation] | None = Field(default=None)


class MidConvSystemBlock(CacheableContentBlock, StandardContentProtocol):
    """System instructions injected at a specific point mid-conversation."""

    type: Literal["mid_conv_system"] = Field(default="mid_conv_system")
    content: list[TextBlock] = Field(description="System instruction text blocks.")


# --------------------------------
# Custom Zylon Blocks
# --------------------------------


class Base64BinarySource(BaseModel):
    """Base64 source payload for arbitrary binary data."""

    type: Literal["base64"]
    data: str = Field(description="Base64-encoded binary data")
    media_type: str = Field(description="MIME type, e.g. 'application/pdf'")

    model_config = ConfigDict(extra="allow")


class URIBinarySource(BaseModel):
    """URI source payload for arbitrary binary data."""

    type: Literal["url"]
    url: str = Field(
        description="Publicly reachable URI",
        validation_alias=AliasChoices("uri", "url"),
        serialization_alias="url",
    )

    model_config = ConfigDict(extra="allow")


BinarySource = Annotated[
    Base64BinarySource | URIBinarySource, Field(discriminator="type")
]


class BinaryBlock(BaseContentBlock, ExtendedContentProtocol):
    """Arbitrary binary payload (PDF, ZIP, …) encoded as base64."""

    type: Literal["binary"] = Field(default="binary")
    filename: str | None = Field(default=None)
    source: BinarySource = Field(description="Binary source payload")

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_payload(cls, values: Any) -> Any:
        return _upgrade_legacy_binary_source_payload(values)

    @classmethod
    def from_bytes(
        cls, binary: bytes, mime_type: str, filename: str | None = None
    ) -> BinaryBlock:
        return cls(
            filename=filename,
            source=Base64BinarySource(
                type="base64",
                data=base64.b64encode(binary).decode(),
                media_type=mime_type,
            ),
        )

    @classmethod
    def from_text(
        cls, text: str, mime_type: str, filename: str | None = None
    ) -> BinaryBlock:
        return cls.from_bytes(text.encode(), mime_type=mime_type, filename=filename)


class LocalResourceBlock(BaseContentBlock, StandardContentProtocol):
    """Reference to a local file produced by code execution."""

    type: Literal["local_resource"] = Field(default="local_resource")
    file_path: str = Field(
        description="Absolute path to the file inside the execution environment"
    )
    file_id: str | None = Field(
        default=None,
        description="Base64url-encoded storage file ID used to download the file via the files API",
    )
    name: str = Field(description="Human-readable file name (stem, without extension)")
    mime_type: str = Field(description="MIME type of the file")


class ResourceLinkBlock(BaseContentBlock, StandardContentProtocol):
    """Reference to an external resource by URI (not embedded)."""

    type: Literal["resource_link"] = Field(default="resource_link")
    uri: str = Field(description="URI of the external resource")
    name: str = Field(description="Human-readable resource name")
    description: str | None = Field(default=None)
    mime_type: str | None = Field(default=None)


class ResourceBlock(BaseContentBlock, StandardContentProtocol):
    """Embedded resource with metadata."""

    class Resource(BaseModel):
        uri: str
        name: str
        description: str | None = None
        mime_type: str | None = None

    type: Literal["resource"] = Field(default="resource")
    resource: Resource = Field(description="Embedded resource metadata")


class SourceBlock(BaseContentBlock, ExtendedContentProtocol):
    """Document chunks surfaced as RAG context attribution."""

    type: Literal["source"] = Field(default="source")
    sources: list[SourceType] = Field(
        description="Document chunks used as context for this response"
    )

    @classmethod
    def from_nodes(cls, nodes: list[NodeWithScore]) -> SourceBlock:
        return cls(sources=[Chunk.from_node(node) for node in nodes])

    @classmethod
    def from_sources(cls, sources: list[SourceType]) -> SourceBlock:
        return cls(sources=sources)


BasicContentBlockType = (
    TextBlock
    | ImageBlock
    | AudioBlock
    | BinaryBlock
    | LocalResourceBlock
    | ResourceLinkBlock
    | ResourceBlock
    | SourceBlock
    | ThinkingBlock
    | RedactedThinkingBlock
    | ToolUseBlock
    | ServerToolUseBlock
    | ContainerUploadBlock
    | DocumentBlock
    | SearchResultBlock
    | MidConvSystemBlock
)


class TLDRBlock(BaseContentBlock, ExtendedContentProtocol):
    """Condensed summary block."""

    type: Literal["tldr"] = Field(default="tldr")
    content: list[BasicContentBlockType] = Field(default_factory=list)
    tldr_side: Literal["left", "right"] = Field(default="left")


ResultContentBlockType = BasicContentBlockType | TLDRBlock
