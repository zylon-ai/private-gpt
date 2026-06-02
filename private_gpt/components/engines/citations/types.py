from dataclasses import dataclass
from typing import Any, Literal

from llama_index.core.schema import MetadataMode, NodeWithScore

from private_gpt.components.chunk.models import Chunk, SourceType, Website
from private_gpt.components.ingest.metadata_helper import (
    MetadataFlags,
    MetadataKeys,
    MetadataNode,
)


@dataclass
class Citation:
    """Represents a citation in a document.

    A citation is a reference to a document in the text.
    It has a start and end position in the text,
    and a value that contains the citation information.

    Attributes:
    - text: the citation text
    - start: the start position of the citation in the text
    - end: the end position of the citation in the text
    - value: the citation information
    """

    text: str
    value: dict[str, Any] | None = None
    doc_id: str | None = None
    artifact_id: str | None = None
    source_id: str | None = None


@dataclass
class Document:
    """Represents a document in the system.

    It corresponds to a document in the database and
    contains information about the document to cite.

    Attributes:
    - id: the document id
    - shorter_id: the shorter document id.
            This is used for citation references to reduce the length of the reference.
    - document_id: the document id
    - text: the text of the document
    """

    id_: str
    shorter_id: str | None
    document_id: str | None

    type: Literal["document", "webpage"]
    text: str

    _metadata: dict[str, Any] | None = None

    @property
    def id(self) -> str:
        """Return the shorter id if available, otherwise return the full id."""
        return self.shorter_id or self.id_

    @classmethod
    def from_node(cls, node: NodeWithScore) -> "Document":
        """Create a Document instance from a node."""
        metadata = node.metadata or {}
        metadata.update(node.metadata)
        metadata.update(
            {
                MetadataNode.SCORE.value: node.score,
            }
        )

        return cls(
            id_=node.id_,
            type="document",
            shorter_id=node.metadata.get(MetadataFlags.SHORTER_ID.value),
            document_id=node.metadata.get(MetadataKeys.ARTIFACT_ID.value),
            text=node.get_content(MetadataMode.LLM),
            _metadata=node.metadata if node.metadata else None,
        )

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "Document":
        """Create a Document instance from a Chunk."""
        metadata = chunk.metadata or {}
        metadata.update(chunk.document.doc_metadata or {})
        metadata.update(
            {
                MetadataNode.SCORE.value: chunk.score,
            }
        )

        return cls(
            id_=chunk.id or str(chunk.document.artifact),
            type="document",
            shorter_id=metadata.get(MetadataFlags.SHORTER_ID.value),
            document_id=chunk.document.artifact,
            text=chunk.text,
            _metadata=chunk.metadata if chunk.metadata else None,
        )

    @classmethod
    def from_webpage(cls, website: Website) -> "Document":
        """Create a Document instance from a webpage."""
        metadata = website.metadata or {}
        metadata.update(website.metadata or {})

        if website.title:
            metadata["title"] = website.title
        if website.description:
            metadata["description"] = website.description

        return cls(
            id_=website.url,
            type="webpage",
            shorter_id=metadata.get(MetadataFlags.SHORTER_ID.value),
            document_id=website.url,
            text=website.content or "",
            _metadata=website.metadata if website.metadata else None,
        )

    @classmethod
    def from_source(cls, source: SourceType) -> "Document":
        """Create a Document instance from a source."""
        match source:
            case Chunk():
                return cls.from_chunk(source)
            case Website():
                return cls.from_webpage(source)
            case _:
                raise ValueError(f"Unsupported source type: {type(source)}")

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the metadata of the document."""
        metadata: dict[str, Any] = self._metadata or {}
        metadata.update(
            {
                MetadataFlags.SHORTER_ID.value: self.shorter_id,
                MetadataKeys.ARTIFACT_ID.value: self.document_id,
            }
        )
        return metadata

    def update_metadata(self, key: str, value: Any) -> None:
        """Update the metadata of the document."""
        if self._metadata is None:
            self._metadata = {}
        self._metadata[key] = value
