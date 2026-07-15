import enum
from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from private_gpt.chat.extensions.context_filter import ContextFilter
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.llm.llm_helper import get_tokenizer
from private_gpt.components.readers.nodes import NodeType
from private_gpt.components.readers.nodes.frozen_node import FrozenNode
from private_gpt.components.readers.nodes.image_node import ImageNode
from private_gpt.components.readers.nodes.list_node import ListItemNode, ListNode
from private_gpt.components.readers.nodes.section_node import SectionNode
from private_gpt.components.readers.nodes.table_node import TableNode, TableRowNode
from private_gpt.components.readers.nodes.text_node import TextNode
from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode, TreeNode
from private_gpt.events.models import ResultContentBlockType
from private_gpt.server.content.content_service import ContentService
from private_gpt.server.utils.auth import authenticated
from private_gpt.server.utils.openapi_models import OpenAPIValidationErrorResponse

NodeTypeName = Literal[
    "ImageNode",
    "ListItemNode",
    "ListNode",
    "SectionNode",
    "TableNode",
    "TableRowNode",
    "TextNode",
]

NODE_TYPE_MAPPING: dict[str, type[NodeType]] = {
    "ImageNode": ImageNode,
    "ListItemNode": ListItemNode,
    "ListNode": ListNode,
    "SectionNode": SectionNode,
    "TableNode": TableNode,
    "TableRowNode": TableRowNode,
    "TextNode": TextNode,
}


class ContentFormat(enum.StrEnum):
    """Enumeration of content retrieval formats."""

    Object = "object"
    Markdown = "markdown"


class ContentTree(BaseModel):
    """Structured representation of document content as a tree."""

    id: str = Field(..., description="Unique identifier of the node")
    type: str = Field(..., description="Type of the node (e.g., section, paragraph)")
    content: str = Field(..., description="Text content of the node")
    children: list["ContentTree"] = Field(
        default_factory=list,
        description="Child nodes representing nested content structure",
    )

    @classmethod
    def from_node(
        cls, node: TreeNode, mode: TreeMetadataMode = TreeMetadataMode.USER
    ) -> "ContentTree":
        """Recursively convert a TreeNode into a ContentTree."""
        node_type = getattr(node, "type", None)
        if not isinstance(node_type, str):
            node_type = node.get_type()
        return cls(
            id=node.id_,
            type=node_type,
            content=node.get_content(mode),
            children=[cls.from_node(child, mode=mode) for child in node.children],
        )


content_router = APIRouter(
    prefix="/v1/artifacts",
    dependencies=[Depends(authenticated)],
    tags=["Artifacts"],
    responses={401: {"description": "Unauthorized"}},
)


class ContentFilter(BaseModel):
    """Filter the content by node types to include in the response."""

    include: list[NodeTypeName] | None = Field(
        default=None,
        description=(
            "List of node types to include in the content response. "
            "If not specified, all node types will be included. "
            "Example node types include TextNode, ImageNode, TableNode, etc."
        ),
    )
    exclude: list[NodeTypeName] | None = Field(
        default=None,
        description=(
            "List of node types to exclude from the content response. "
            "If not specified, no node types will be excluded. "
            "Example node types include TextNode, ImageNode, TableNode, etc."
        ),
    )
    node_ids: list[str] | None = Field(
        default=None,
        description=(
            "List of specific node IDs to retrieve from the document tree. "
            "When specified, only these nodes (and optionally their children) will be returned. "
            "Useful for retrieving specific sections or parts of a document. "
            "Example: ['382b0aab-3c63-44a1-ae2e-1ee234009d6e', '6d2a3086-10bc-4d76-885b-2208c211b648']"
        ),
    )
    include_children: bool = Field(
        default=True,
        description=(
            "When node_ids is specified, determines whether to include the full subtree "
            "below each selected node. If True (default), returns complete subtrees. "
            "If False, returns only the specified nodes without their descendants."
        ),
    )
    include_ancestors: bool = Field(
        default=False,
        description=(
            "When node_ids is specified, determines whether to include ancestor nodes "
            "in the path from each selected node to the document root. "
            "If True, provides structural context. If False (default), returns only selected subtrees."
        ),
    )

    def get_include_types(self) -> list[type[NodeType]] | None:
        """Convert node type names to actual type classes for include filter."""
        if self.include is None:
            return None
        return [NODE_TYPE_MAPPING[name] for name in self.include]

    def get_exclude_types(self) -> list[type[NodeType]] | None:
        """Convert node type names to actual type classes for exclude filter."""
        if self.exclude is None:
            return None
        return [NODE_TYPE_MAPPING[name] for name in self.exclude]


class ContentBody(BaseModel):
    """Request body for retrieving full document content with filtering options."""

    context_filter: ContextFilter = Field(
        ...,
        description=(
            "Filter to select documents to retrieve. "
            "Supports filtering by collection, artifacts, and metadata."
        ),
    )
    format: ContentFormat = Field(
        default=ContentFormat.Markdown,
        description=(
            "Format for returned content. 'object' returns structured data, "
            "'markdown' returns content formatted as markdown text."
        ),
    )
    filter: ContentFilter | None = Field(
        default=None,
        description=(
            "Content filtering options to include or exclude specific node types. "
            "Use this to control the types of content returned in the response."
        ),
    )
    max_tokens: int | None = Field(
        None,
        description="Maximum number of tokens to return in the content. If not set, returns full content of the documents.",
        ge=1,
    )


class ContentDocumentResponse(BaseModel):
    """Individual document response with full content."""

    artifact_id: str = Field(..., description="Identifier of the document")
    content: str | ContentTree = Field(
        ..., description="Full text content of the document"
    )


class ContentResponse(BaseModel):
    """Response containing full document content for filtered documents."""

    data: list[ContentDocumentResponse] = Field(
        ..., description="List of documents with their full content"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data": [
                        {
                            "artifact_id": "annual_report_2023",
                            "content": "ANNUAL REPORT 2023\n\nExecutive Summary\n\nFiscal year 2023 marked a transformative period...",
                        },
                    ]
                },
            ]
        }
    }


class ChunkedContentDocumentResponse(BaseModel):
    """Response model for chunked document content."""

    artifact_id: str = Field(..., description="Identifier of the document")
    content: list[ResultContentBlockType] = Field(
        ...,
        description="Chunked content of the document, split into manageable pieces",
    )


class ChunkedContentResponse(BaseModel):
    """Response containing chunked document content for chat usage."""

    data: list[ChunkedContentDocumentResponse] = Field(
        ...,
        description="List of documents with their content split into chunks for chat usage",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data": [
                        {
                            "artifact_id": "annual_report_2023",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "ANNUAL REPORT 2023\n\nExecutive Summary\n\nFiscal year 2023 marked a transformative period...",
                                },
                            ],
                        },
                    ]
                },
            ]
        }
    }


@content_router.post(
    "/content",
    response_model=ContentResponse,
    summary="Retrieve Full Document Content",
    responses={
        200: {
            "description": "Successful content retrieval",
            "content": {
                "application/json": {
                    "examples": {
                        "markdown_format": {
                            "summary": "Complete document content in markdown format",
                            "value": {
                                "data": [
                                    {
                                        "artifact_id": "annual_report_2023",
                                        "content": "ANNUAL REPORT 2023\n\nExecutive Summary\n\nFiscal year 2023 marked a transformative period for our organization with record-breaking performance across all key metrics...\n\n[Full document content continues...]",
                                    },
                                    {
                                        "artifact_id": "quarterly_summary_q4",
                                        "content": "Q4 QUARTERLY SUMMARY\n\nQuarter Overview\n\nThe fourth quarter concluded our strongest year on record, with revenue growth of 34% year-over-year...\n\n[Full document content continues...]",
                                    },
                                ]
                            },
                        },
                        "object_format": {
                            "summary": "Structured document content as tree objects",
                            "value": {
                                "data": [
                                    {
                                        "artifact_id": "annual_report_2023",
                                        "content": "ANNUAL REPORT 2023\n\nExecutive Summary",
                                    }
                                ]
                            },
                        },
                        "filtered_content": {
                            "summary": "Documents with node type filtering",
                            "value": {
                                "data": [
                                    {
                                        "artifact_id": "finance_report_dec",
                                        "content": "DECEMBER FINANCIAL REPORT\n\nRevenue Analysis\n\nDecember revenue reached $850K, representing a 12% increase from November...",
                                    }
                                ]
                            },
                        },
                        "subtree_retrieval": {
                            "summary": "Specific sections retrieved by node IDs",
                            "value": {
                                "data": [
                                    {
                                        "artifact_id": "annual_report_2023",
                                        "content": "Introduction\n\nKey Players in the Streaming Wars",
                                    }
                                ]
                            },
                        },
                    }
                }
            },
        },
        422: {
            "model": OpenAPIValidationErrorResponse,
            "description": "Validation Error - Invalid request parameters",
            "content": {
                "application/json": {
                    "examples": {
                        "missing_context_filter": {
                            "summary": "Missing context filter",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "context_filter"],
                                        "msg": "field required",
                                        "type": "value_error.missing",
                                    }
                                ]
                            },
                        },
                        "invalid_collection": {
                            "summary": "Invalid collection reference",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "context_filter", "collection"],
                                        "msg": "Collection 'nonexistent' not found",
                                        "type": "value_error",
                                    }
                                ]
                            },
                        },
                    }
                }
            },
        },
    },
    tags=["Artifacts"],
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ContentBody"}
                }
            },
            "required": True,
            "description": (
                "Request body for retrieving full document content.\n\n"
                "Contains context filtering options to select specific "
                "documents by collection, artifacts, and metadata criteria. "
                "Unlike chunk retrieval, this returns complete document "
                "content rather than excerpts.\n\n"
                "Format Options:\n"
                "* markdown (default): Returns content as formatted markdown text\n"
                "* object: Returns structured tree representation of document hierarchy\n\n"
                "Content Filtering:\n"
                "* include: Specify node types to include (TextNode, ImageNode, TableNode, etc.)\n"
                "* exclude: Specify node types to exclude from the response\n"
                "* node_ids: Specify exact node IDs to retrieve specific sections or parts\n"
                "* include_children: Whether to include full subtrees below selected nodes\n"
                "* include_ancestors: Whether to include ancestor path for context\n\n"
                "The request body defines filtering criteria for bulk "
                "document content retrieval from the ingested knowledge base."
            ),
        }
    },
)
def content_retrieval(request: Request, body: ContentBody) -> ContentResponse:
    """Retrieve the full content of filtered documents.

    This endpoint provides access to complete document content rather than
    semantic chunks. Use this when you need the entire context of documents
    rather than relevant excerpts.

    Key Features:
    * Full Content: Get complete documents rather than chunks
    * Multiple Formats: Return as markdown text or structured objects
    * Node Type Filtering: Include or exclude images, tables, and other node types
    * Node ID Filtering: Retrieve specific sections by their node IDs
    * Filtered Retrieval: Select specific documents using metadata filters
    * Bulk Retrieval: Get multiple documents in one request

    Format Options:
    * markdown (default): Returns flattened markdown text representation
    * object: Returns hierarchical tree structure with typed nodes

    Content Filtering:
    * Use include to retrieve only specific node types
    * Use exclude to omit unwanted content types
    * Use node_ids to retrieve specific sections or nodes by ID
    * Supports TextNode, ImageNode, TableNode, and other node types

    Subtree Filtering:
    * Specify node_ids to retrieve only certain sections/nodes
    * Set include_children=True (default) to get full subtrees
    * Set include_children=False to get only specified nodes
    * Set include_ancestors=True to include parent context

    Notes:
    * Node type filtering is applied to all retrieved documents
    * Object format preserves document structure and hierarchy
    * Markdown format provides a flattened, readable text representation
    * When node_ids is specified, returns filtered tree structure
    """
    service: ContentService = request.state.injector.get(ContentService)

    responses: list[ContentDocumentResponse] = []
    mode = (
        TreeMetadataMode.USER
        if body.format == ContentFormat.Markdown
        else TreeMetadataMode.NONE
    )

    if not body.filter:
        body.filter = ContentFilter()

    for artifact_id, root_node in service.retrieve_document_content(
        context_filter=body.context_filter,
        include=body.filter.get_include_types(),
        exclude=body.filter.get_exclude_types(),
        node_ids=body.filter.node_ids,
        include_children=body.filter.include_children,
        include_ancestors=body.filter.include_ancestors,
    ):
        if body.format == ContentFormat.Object:
            tree_object = ContentTree.from_node(root_node, mode=mode)
            responses.append(
                ContentDocumentResponse(artifact_id=artifact_id, content=tree_object)
            )
        else:
            frozen_node = FrozenNode.from_node(root_node, modes=[mode])
            responses.append(
                ContentDocumentResponse(
                    artifact_id=artifact_id, content=frozen_node.get_content(mode)
                )
            )
            del frozen_node

    return ContentResponse(data=responses)


@content_router.post(
    "/chunked-content",
    response_model=ChunkedContentResponse,
    summary="Retrieve Full Document Content in Chunks",
    responses={
        200: {
            "description": "Successful chunked content retrieval",
            "content": {
                "application/json": {
                    "examples": {
                        "chunked_documents": {
                            "summary": "Complete documents split into chat chunks",
                            "value": {
                                "data": [
                                    {
                                        "artifact_id": "annual_report_2023",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": "ANNUAL REPORT 2023\n\nExecutive Summary\n\nFiscal year 2023 marked a transformative period for our organization with record-breaking performance across all key metrics.",
                                            },
                                            {
                                                "type": "text",
                                                "text": "Revenue Analysis\n\nTotal revenue reached $12.4M, representing a 34% increase year-over-year. Growth was driven primarily by enterprise customer acquisition and subscription renewals.",
                                            },
                                        ],
                                    },
                                    {
                                        "artifact_id": "quarterly_summary_q4",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": "Q4 QUARTERLY SUMMARY\n\nQuarter Overview\n\nThe fourth quarter concluded our strongest year on record, with significant achievements in customer satisfaction and operational efficiency.",
                                            }
                                        ],
                                    },
                                ]
                            },
                        },
                    }
                }
            },
        },
        422: {
            "model": OpenAPIValidationErrorResponse,
            "description": "Validation Error - Invalid request parameters",
            "content": {
                "application/json": {
                    "examples": {
                        "missing_context_filter": {
                            "summary": "Missing context filter",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "context_filter"],
                                        "msg": "field required",
                                        "type": "value_error.missing",
                                    }
                                ]
                            },
                        },
                        "invalid_max_tokens": {
                            "summary": "Invalid max tokens parameter",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "max_tokens"],
                                        "msg": "ensure this value is greater than or equal to 1",
                                        "type": "value_error.number.not_ge",
                                    }
                                ]
                            },
                        },
                        "invalid_collection": {
                            "summary": "Invalid collection reference",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "context_filter", "collection"],
                                        "msg": "Collection 'nonexistent' not found",
                                        "type": "value_error",
                                    }
                                ]
                            },
                        },
                    }
                }
            },
        },
    },
    tags=["Artifacts"],
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ContentBody"}
                }
            },
            "required": True,
            "description": (
                "Request body for retrieving chunked document content.\n\n"
                "Contains context filtering options to select specific "
                "documents by collection, artifacts, and metadata criteria, "
                "plus optional token limiting for chat optimization.\n\n"
                "Content Filtering:\n"
                "* include: Specify node types to include (TextNode, ImageNode, TableNode, etc.)\n"
                "* exclude: Specify node types to exclude from the response\n\n"
                "Token Management:\n"
                "* max_tokens: Optional parameter to limit total tokens in response\n"
                "* Prevents context window overflow in chat interfaces\n\n"
                "The request body defines filtering criteria and chunking "
                "parameters for retrieving documents split into chat-ready "
                "content blocks with citations."
            ),
        }
    },
)
async def chunked_content_retrieval(
    request: Request, body: ContentBody
) -> ChunkedContentResponse:
    """Retrieve full document content split into chat-optimized chunks.

    This endpoint provides access to complete document content split into
    manageable chunks suitable for chat interfaces. Unlike semantic chunk
    retrieval, this returns complete documents divided sequentially.

    Key Features:
    * Chat-Optimized Chunking: Documents split into conversational pieces
    * Node Type Filtering: Include or exclude images, tables, and other node types
    * Token-Aware Splitting: Respects token limits for chat context management
    * Sequential Chunks: Maintains document order and narrative flow
    * Filtered Retrieval: Select specific documents using metadata filters
    * Token Limiting: Optional max_tokens parameter to control response size

    Content Filtering:
    * Use include to retrieve only specific node types
    * Use exclude to omit unwanted content types
    * Supports TextNode, ImageNode, TableNode, and other node types
    * Filtering is applied before chunking

    Chunking Process:
    1. Retrieve filtered documents based on context criteria
    2. Apply node type filters (include/exclude)
    3. Split documents into chat-appropriate segments respecting max_tokens
    4. Return structured chunks with metadata and citations

    Notes:
    * Chunks maintain document structure and logical flow
    * Token limiting prevents context window overflow
    * Node type filtering reduces payload size and improves relevance
    * Use `/artifacts/search` endpoint for semantic search instead
    """
    service: ContentService = request.state.injector.get(ContentService)
    llm_component = request.state.injector.get(LLMComponent)

    if not body.filter:
        body.filter = ContentFilter()

    max_length = body.max_tokens or llm_component.metadata().context_window
    content_blocks = [
        (artifact_id, content_blocks)
        async for artifact_id, content_blocks in service.retrieve_chunked_document_content(
            context_filter=body.context_filter,
            include=body.filter.get_include_types(),
            exclude=body.filter.get_exclude_types(),
            max_length=max_length - (max_length // 10) if max_length else None,
            tokenizer_fn=get_tokenizer(),
            generate_citations=True,
        )
    ]
    return ChunkedContentResponse(
        data=[
            ChunkedContentDocumentResponse(
                artifact_id=artifact_id,
                content=content,
            )
            for artifact_id, content in content_blocks
        ]
    )
