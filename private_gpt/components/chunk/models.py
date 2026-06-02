import uuid
from typing import Any, Literal

from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel, Field

from private_gpt.components.ingest.metadata_helper import (
    MetadataChunk,
    MetadataDocument,
    MetadataKeys,
)
from private_gpt.components.readers.nodes import TreeNode
from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode
from private_gpt.components.web.web_search.models import WebSearchResult
from private_gpt.server.ingest.model import IngestedDoc


class Chunk(BaseModel):
    """Represents a chunk of text content from an ingested document."""

    object: Literal["context.chunk"] = Field(
        description="Object type identifier, always 'context.chunk' for chunk responses"
    )
    id: str | None = Field(
        default=None,
        description="Unique identifier for the chunk within the document",
        examples=["chunk_123e4567-e89b-12d3-a456-426614174000", "doc_page_1_chunk_3"],
    )
    score: float = Field(
        description="Relevance score indicating how well this chunk matches the query (0.0 to 1.0, higher is better)",
        # We need to set a close number to avoid mantissa errors
        ge=-0.01,
        le=1.01,
        examples=[0.023, 0.856, 0.342],
    )
    document: IngestedDoc = Field(
        description="Reference to the parent document containing metadata and ingestion information"
    )
    text: str = Field(
        description="The actual text content of the chunk extracted from the document",
        examples=[
            "Outbound sales increased 20%, driven by new leads.",
            "The quarterly financial report shows significant growth in the technology sector.",
            "Avatar is set in an Asian and Arctic-inspired world where some people can manipulate elements.",
        ],
    )
    content_type: str = Field(
        default="text/plain",
        description="MIME type indicating the format of the chunk content",
        examples=["text/plain", "text/html", "text/markdown", "application/json"],
    )
    previous_texts: list[str] | None = Field(
        default=None,
        description="List of text chunks that appear before this chunk in the document, providing preceding context",
        examples=[
            ["SALES REPORT 2023", "Inbound didn't show major changes."],
            ["Chapter 1: Introduction", "Our company mission is to innovate."],
            None,
        ],
    )
    next_texts: list[str] | None = Field(
        default=None,
        description="List of text chunks that appear after this chunk in the document, providing following context",
        examples=[
            [
                "New leads came from Google Ads campaign.",
                "The campaign was run by the Marketing Department",
            ],
            [
                "The next quarter will focus on customer retention.",
                "Budget allocation has been approved.",
            ],
            None,
        ],
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata about the chunk including positioning information and document properties",
        examples=[
            {
                "title": "Sales Report 2023",
                "author": "John Doe",
                "date": "2023-01-01",
                "abs_idx": 5,
                "rel_idx": 2,
            },
            {
                "file_name": "quarterly_report.pdf",
                "page_number": 3,
                "section": "Financial Overview",
                "abs_idx": 12,
                "rel_idx": 0,
            },
            None,
        ],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "object": "context.chunk",
                    "id": "chunk_123e4567-e89b-12d3-a456-426614174000",
                    "score": 0.856,
                    "document": {
                        "object": "ingest.document",
                        "artifact": "quarterly_report_q3",
                        "doc_metadata": {
                            "file_name": "Q3_Financial_Report.pdf",
                            "page_number": 5,
                            "department": "finance",
                        },
                    },
                    "text": "Revenue increased by 15% compared to the previous quarter, primarily driven by strong performance in the technology sector.",
                    "content_type": "text/plain",
                    "previous_texts": [
                        "Q3 FINANCIAL SUMMARY",
                        "This report covers the third quarter performance metrics.",
                    ],
                    "next_texts": [
                        "The technology sector contributed 60% of total growth.",
                        "Marketing expenses remained within budget projections.",
                    ],
                    "metadata": {
                        "title": "Q3 Financial Report",
                        "author": "Finance Team",
                        "date": "2023-10-15",
                        "abs_idx": 8,
                        "rel_idx": 3,
                        "section": "Revenue Analysis",
                    },
                }
            ]
        }
    }

    @classmethod
    def from_node(cls: type["Chunk"], node: NodeWithScore) -> "Chunk":
        """Create a Chunk instance from a NodeWithScore object."""
        metadata = {k: v for k, v in node.metadata.items() if k in list(MetadataChunk)}
        if MetadataChunk.ABS_IDX not in metadata:
            abs_idx = node.node.abs_idx if isinstance(node.node, TreeNode) else 0
            metadata[MetadataChunk.ABS_IDX] = abs_idx

        if MetadataChunk.REL_IDX not in metadata:
            idx = node.node.idx if isinstance(node.node, TreeNode) else 0
            metadata[MetadataChunk.REL_IDX] = idx

        return cls(
            object="context.chunk",
            id=node.node.id_,
            score=max(0.0, node.score or 0.0),
            document=IngestedDoc(
                object="ingest.document",
                artifact=str(node.metadata.get(MetadataKeys.ARTIFACT_ID.value)),
                doc_metadata={
                    k: v
                    for k, v in node.metadata.items()
                    if k in list(MetadataDocument)
                },
            ),
            text=(
                node.node.get_content(TreeMetadataMode.USER)
                if isinstance(node.node, TreeNode)
                else node.node.get_content()
            ),
            content_type=node.node.mimetype
            if hasattr(node.node, "mimetype")
            else "text/markdown",
            previous_texts=list(node.metadata.get("previous_texts", [])),
            next_texts=list(node.metadata.get("next_texts", [])),
            metadata=metadata,
        )


class Website(BaseModel):
    """Represents a website URL source."""

    id: str = Field(
        description="Unique identifier for the website source",
        examples=["website_123e4567-e89b-12d3-a456-426614174000"],
    )

    object: Literal["context.website"] = Field(
        description="Object type identifier, always 'context.website' for website sources"
    )
    url: str = Field(
        description="The URL of the website",
        examples=[
            "https://www.example.com",
            "https://docs.privategpt.com/getting-started",
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
        ],
    )

    favicon_url: str | None = Field(
        default=None,
        description="The URL of the website's favicon",
        examples=[
            "https://www.example.com/favicon.ico",
            "https://docs.privategpt.com/favicon.png",
            "https://en.wikipedia.org/static/favicon/wikipedia.ico",
        ],
    )

    title: str | None = Field(
        default=None,
        description="The title of the website or webpage",
        examples=[
            "Example Domain",
            "Getting Started with PrivateGPT - Documentation",
            "Artificial Intelligence - Wikipedia",
        ],
    )
    description: str | None = Field(
        default=None,
        description="A brief description or summary of the website content",
        examples=[
            "This domain is for use in illustrative examples in documents.",
            "PrivateGPT is an open-source project that enables private AI interactions.",
            "Artificial intelligence (AI) is intelligence demonstrated by machines.",
        ],
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata about the website source",
        examples=[
            {
                "accessed_date": "2024-01-15",
                "language": "en",
            },
            {
                "accessed_date": "2024-02-20",
                "language": "fr",
            },
            None,
        ],
    )

    content_type: str | None = Field(
        default=None,
        description="MIME type indicating the format of the website content",
        examples=["text/html", "application/json"],
    )
    content: str | None = Field(
        default=None,
        description="The actual text content extracted from the website",
        examples=[
            "<html><head><title>Example Domain</title></head><body>This domain is for use in illustrative examples in documents.</body></html>",
            "PrivateGPT is an open-source project that enables private AI interactions.",
            "Artificial intelligence (AI) is intelligence demonstrated by machines.",
        ],
    )

    @classmethod
    def from_website_result(cls, result: WebSearchResult) -> "Website":
        """Create a Website instance from a WebSearchResult object."""
        return cls(
            id="website_" + str(uuid.uuid4()),
            object="context.website",
            url=result.url,
            favicon_url=result.favicon_url,
            title=result.title,
            description=result.description,
            content_type=result.content_type,
            content=result.content,
            metadata=result.metadata,
        )


SourceType = Chunk | Website
