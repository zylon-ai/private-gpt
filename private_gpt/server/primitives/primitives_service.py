from collections.abc import Mapping
from typing import Annotated, Any, Literal

from injector import inject, singleton
from pydantic import BaseModel, Field

from private_gpt.chat.extensions.context_filter import ContextFilter
from private_gpt.components.chunk.models import Chunk
from private_gpt.server.primitives.semantic_search_service import SemanticSearchService


class SemanticSearch(BaseModel):
    """Represents a semantic search operation."""

    type: Literal["semantic_search"] = Field(
        default="semantic_search",
        description="Type of search operation, always 'semantic_search' for semantic searches",
    )
    text: str = Field(
        ...,
        description="The text query to find relevant chunks",
        examples=["Q3 2023 sales"],
    )
    context_filter: ContextFilter = Field(
        ...,
        description=(
            "Filter to select specific context from ingested documents. "
            "Can filter by collection, artifacts, and metadata."
        ),
    )
    limit: int = Field(
        default=10,
        description="Maximum number of chunks to return",
        ge=1,
    )
    score_threshold: float = Field(
        default=0.0,
        description="Minimum similarity score threshold for returned chunks",
        ge=0.0,
        le=1.0,
    )
    expand: bool = Field(
        default=False,
        description="Whether to include adjacent chunks for more context",
    )
    check: bool = Field(
        default=True,
        description="Whether to validate the existence of required indexes for the specified artifacts",
    )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: Mapping[str, Any], handler: Any
    ) -> dict[str, Any]:
        json_schema: dict[str, Any] = handler(core_schema)
        # Remove the 'validate' field from the OpenAPI schema
        json_schema.get("properties", {}).pop("check", None)
        # Also remove from required fields if present
        if "required" in json_schema and "check" in json_schema["required"]:
            json_schema["required"].remove("check")
        return json_schema

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "semantic_search",
                "text": "Q3 2023 sales performance",
                "context_filter": {"collection": "reports"},
                "limit": 10,
                "score_threshold": 0.25,
                "expand": True,
            },
            "examples": [
                {
                    "type": "semantic_search",
                    "text": "Q3 2023 sales performance",
                    "context_filter": {"collection": "reports"},
                    "limit": 10,
                    "score_threshold": 0.25,
                    "expand": True,
                }
            ],
        }
    }


class KeywordSearch(BaseModel):
    """Represents a keyword-based search operation."""

    type: Literal["keywords_search"] = Field(
        default="keywords_search",
        description="Type of search operation, always 'keywords_search' for keyword searches",
    )
    keywords: list[str] = Field(
        ...,
        description="List of keywords to find relevant chunks",
        examples=[["sales", "Q3", "2023"]],
    )
    context_filter: ContextFilter = Field(
        ...,
        description=(
            "Filter to select specific context from ingested documents. "
            "Can filter by collection, artifacts, and metadata."
        ),
    )
    limit: int = Field(
        default=10, description="Maximum number of chunks to return", ge=1, le=100
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "keywords_search",
                "keywords": ["sales", "Q3", "2023"],
                "context_filter": {"artifacts": ["q3_report"]},
                "limit": 5,
            },
            "examples": [
                {
                    "type": "keywords_search",
                    "keywords": ["sales", "Q3", "2023"],
                    "context_filter": {"artifacts": ["q3_report"]},
                    "limit": 5,
                }
            ],
        }
    }


class HybridSearch(BaseModel):
    """Represents a hybrid search operation combining semantic and keyword search."""

    type: Literal["hybrid_search"] = Field(
        default="hybrid_search",
        description="Type of search operation, always 'hybrid_search' for combined searches",
    )
    text: str = Field(
        ...,
        description="The text query to find relevant chunks",
        examples=["Q3 2023 sales"],
    )
    keywords: list[str] = Field(
        ...,
        description="List of keywords to find relevant chunks",
        examples=[["sales", "Q3", "2023"]],
    )
    context_filter: ContextFilter = Field(
        ...,
        description=(
            "Filter to select specific context from ingested documents. "
            "Can filter by collection, artifacts, and metadata."
        ),
    )
    limit: int = Field(
        default=10, description="Maximum number of chunks to return", ge=1, le=100
    )
    expand: bool = Field(
        default=False,
        description="Whether to include adjacent chunks for more context",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "hybrid_search",
                "text": "quarterly sales analysis",
                "keywords": ["revenue", "growth", "metrics"],
                "context_filter": {"collection": "financial"},
                "limit": 15,
                "expand": False,
            },
            "examples": [
                {
                    "type": "hybrid_search",
                    "text": "quarterly sales analysis",
                    "keywords": ["revenue", "growth", "metrics"],
                    "context_filter": {"collection": "financial"},
                    "limit": 15,
                    "expand": False,
                }
            ],
        }
    }


SearchBody = Annotated[
    SemanticSearch | KeywordSearch | HybridSearch, Field(discriminator="type")
]

DataSearchResponse = Annotated[
    Chunk,  # add more specific type if needed
    Field(
        discriminator="object",
        description="Represent a list of results from a search operation",
    ),
]


class SearchResponse(BaseModel):
    """Response containing semantically relevant document chunks."""

    object: Literal["list"] = Field(
        default="list", description="Response object type identifier"
    )
    model: Literal["private-gpt"] = Field(
        default="private-gpt", description="Model identifier used for chunk retrieval"
    )
    data: list[DataSearchResponse] = Field(
        ..., description="List of relevant chunks with their metadata and scores"
    )


@singleton
class PrimitivesService:
    @inject
    def __init__(
        self,
        semantic_search_service: SemanticSearchService,
    ) -> None:
        self.semantic_search_service = semantic_search_service

    def search(
        self,
        search: SearchBody,
    ) -> SearchResponse:
        """Perform a semantic search based on the provided query and context filter."""
        match search:
            case SemanticSearch():
                chunks = self.semantic_search_service.retrieve_semantic_relevant(
                    text=search.text,
                    context_filter=search.context_filter,
                    limit=search.limit,
                    expand=search.expand,
                    score_threshold=search.score_threshold,
                    validate=search.check,
                )
                return SearchResponse(
                    data=chunks,
                )
            case _:
                raise ValueError(f"Unsupported search type: {type(search)}")
