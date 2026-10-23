from typing import Annotated, cast

from fastapi import APIRouter, Body, Depends, Request
from fastapi.openapi.models import Example

from private_gpt.server.primitives.primitives_service import (
    PrimitivesService,
    SearchBody,
    SearchResponse,
)
from private_gpt.server.utils.auth import authenticated

primitives_router = APIRouter(
    prefix="/v1/primitives",
    dependencies=[Depends(authenticated)],
    tags=["Primitives"],
    responses={401: {"description": "Unauthorized"}},
)

SEARCH_REQUEST_EXAMPLES = cast(
    dict[str, Example],
    {
        "semantic_search": {
            "summary": "Semantic Search Example",
            "value": {
                "type": "semantic_search",
                "text": "Q3 2023 sales performance",
                "context_filter": {"collection": "reports"},
                "limit": 10,
                "expand": True,
            },
        },
        "keyword_search": {
            "summary": "Keyword Search Example (Coming Soon)",
            "value": {
                "type": "keywords_search",
                "keywords": ["sales", "Q3", "2023"],
                "context_filter": {"artifacts": ["q3_report"]},
                "limit": 5,
            },
        },
        "hybrid_search": {
            "summary": "Hybrid Search Example (Coming Soon)",
            "value": {
                "type": "hybrid_search",
                "text": "quarterly sales analysis",
                "keywords": ["revenue", "growth", "metrics"],
                "context_filter": {"collection": "financial"},
                "limit": 15,
                "expand": False,
            },
        },
    },
)


@primitives_router.post(
    "/search",
    response_model=None,
    summary="Retrieve Chunks using Semantic, Keyword, or Hybrid Search",
    responses={
        200: {
            "description": "Successful chunk retrieval",
            "content": {
                "application/json": {
                    "examples": {
                        "semantic_search_results": {
                            "summary": "Semantic search with chunks",
                            "value": {
                                "object": "list",
                                "model": "private-gpt",
                                "data": [
                                    {
                                        "object": "context.chunk",
                                        "score": 0.89,
                                        "document": {
                                            "object": "ingest.document",
                                            "artifact": "q3_report",
                                            "doc_metadata": {
                                                "title": "Q3 2023 Report",
                                                "file_name": "Q3_Sales.pdf",
                                            },
                                        },
                                        "text": "Q3 sales increased by 25% compared to Q2, driven primarily by new customer acquisitions in the enterprise segment.",
                                        "previous_texts": [
                                            "Q2 comparison shows steady growth trends."
                                        ],
                                        "next_texts": [
                                            "Regional breakdown indicates strongest performance in North America."
                                        ],
                                    },
                                    {
                                        "object": "context.chunk",
                                        "score": 0.76,
                                        "document": {
                                            "object": "ingest.document",
                                            "artifact": "q3_report",
                                            "doc_metadata": {
                                                "title": "Q3 2023 Report",
                                                "page_number": 5,
                                            },
                                        },
                                        "text": "Sales performance metrics exceeded targets across all key verticals during the third quarter.",
                                        "previous_texts": None,
                                        "next_texts": None,
                                    },
                                ],
                            },
                        },
                        "keyword_search_results": {
                            "summary": "Keyword-based search results",
                            "value": {
                                "object": "list",
                                "model": "private-gpt",
                                "data": [
                                    {
                                        "object": "context.chunk",
                                        "score": 0.95,
                                        "document": {
                                            "object": "ingest.document",
                                            "artifact": "financial_docs",
                                            "doc_metadata": {
                                                "department": "finance",
                                                "quarter": "Q3",
                                            },
                                        },
                                        "text": "Q3 2023 sales figures show significant growth in revenue streams.",
                                        "previous_texts": None,
                                        "next_texts": None,
                                    }
                                ],
                            },
                        },
                        "hybrid_search_results": {
                            "summary": "Combined semantic and keyword search",
                            "value": {
                                "object": "list",
                                "model": "private-gpt",
                                "data": [
                                    {
                                        "object": "context.chunk",
                                        "score": 0.91,
                                        "document": {
                                            "object": "ingest.document",
                                            "artifact": "annual_report",
                                            "doc_metadata": {"year": "2023"},
                                        },
                                        "text": "Annual sales performance demonstrates consistent growth with Q3 being the strongest quarter.",
                                        "previous_texts": [
                                            "Market analysis indicates favorable conditions.",
                                        ],
                                        "next_texts": [
                                            "Strategic initiatives continue to drive results.",
                                        ],
                                    }
                                ],
                            },
                        },
                    }
                }
            },
        },
        422: {
            "description": "Validation Error - Invalid request parameters",
            "content": {
                "application/json": {
                    "examples": {
                        "invalid_search_type": {
                            "summary": "Unsupported search type",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "type"],
                                        "msg": "Invalid search type. Supported types: semantic_search, keywords_search, hybrid_search",
                                        "type": "value_error",
                                    }
                                ]
                            },
                        },
                        "invalid_limit": {
                            "summary": "Limit parameter out of range",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "limit"],
                                        "msg": "ensure this value is less than or equal to 100",
                                        "type": "value_error.number.not_le",
                                    }
                                ]
                            },
                        },
                        "empty_text_query": {
                            "summary": "Missing text query for semantic or hybrid search",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "text"],
                                        "msg": "field required",
                                        "type": "value_error.missing",
                                    }
                                ]
                            },
                        },
                        "empty_keywords": {
                            "summary": "Missing keywords for keyword or hybrid search",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "keywords"],
                                        "msg": "field required",
                                        "type": "value_error.missing",
                                    }
                                ]
                            },
                        },
                        "invalid_context_filter": {
                            "summary": "Invalid context filter format",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "context_filter"],
                                        "msg": "Invalid context filter configuration",
                                        "type": "value_error",
                                    }
                                ]
                            },
                        },
                    }
                }
            },
        },
        501: {
            "description": "Not Implemented - Search type not yet supported",
            "content": {
                "application/json": {
                    "examples": {
                        "feature_not_implemented": {
                            "summary": "Keyword or hybrid search not yet implemented",
                            "value": {
                                "detail": "Search type 'keywords_search' or 'hybrid_search' is not yet implemented. Currently only 'semantic_search' is supported."
                            },
                        }
                    }
                }
            },
        },
    },
    tags=["Primitives"],
    openapi_extra={
        "requestBody": {
            "required": True,
            "description": (
                "Request body for searching document chunks using different strategies.\n\n"
                "Supports three search types:\n"
                "- **semantic_search**: Find chunks based on semantic similarity to the text query\n"
                "- **keywords_search**: Find chunks containing specific keywords (coming soon)\n"
                "- **hybrid_search**: Combine semantic and keyword matching (coming soon)\n\n"
                "The request body defines search criteria, context filtering options, "
                "result limits, and expansion settings for retrieving relevant document chunks."
            ),
            "content": {
                "application/json": {
                    "examples": SEARCH_REQUEST_EXAMPLES,
                }
            },
        },
        "x-fern-examples": [
            {
                "name": "Semantic search",
                "request": {
                    "type": "semantic_search",
                    "text": "Q3 2023 sales performance",
                    "context_filter": {"collection": "reports"},
                    "limit": 10,
                    "expand": True,
                },
                "response": {
                    "body": {
                        "object": "list",
                        "model": "private-gpt",
                        "data": [
                            {
                                "object": "context.chunk",
                                "score": 0.89,
                                "document": {
                                    "object": "ingest.document",
                                    "artifact": "q3_report",
                                    "doc_metadata": {
                                        "title": "Q3 2023 Report",
                                        "file_name": "Q3_Sales.pdf",
                                    },
                                },
                                "text": "Q3 sales increased by 25% compared to Q2, driven primarily by new customer acquisitions in the enterprise segment.",
                                "previous_texts": [
                                    "Q2 comparison shows steady growth trends."
                                ],
                                "next_texts": [
                                    "Regional breakdown indicates strongest performance in North America."
                                ],
                            }
                        ],
                    }
                },
            }
        ],
    },
)
def search(
    request: Request,
    body: Annotated[
        SearchBody,
        Body(
            description=(
                "Request body for searching document chunks using different strategies.\n\n"
                "Supports three search types:\n"
                "- **semantic_search**: Find chunks based on semantic similarity to the text query\n"
                "- **keywords_search**: Find chunks containing specific keywords (coming soon)\n"
                "- **hybrid_search**: Combine semantic and keyword matching (coming soon)\n\n"
                "The request body defines search criteria, context filtering options, "
                "result limits, and expansion settings for retrieving relevant document chunks."
            ),
            openapi_examples=SEARCH_REQUEST_EXAMPLES,
        ),
    ],
) -> SearchResponse:
    """Perform document chunk search using semantic, keyword, or hybrid strategies.

    This endpoint provides flexible search capabilities across ingested documents
    with support for different search strategies based on use case requirements.

    Search Types:
    * **Semantic Search**: Uses vector embeddings to find chunks with similar meaning
      to the provided text query, regardless of exact keyword matches
    * **Keyword Search**: Finds chunks containing specific keywords with exact or
      fuzzy matching capabilities (implementation pending)
    * **Hybrid Search**: Combines semantic similarity with keyword matching for
      comprehensive results (implementation pending)

    Key Features:
    * **Score-based Ranking**: Results include similarity/relevance scores
    * **Context Filtering**: Narrow search to specific collections,
    artifacts, or metadata
    * **Adjacent Context**: Optionally retrieve surrounding chunks for richer context
    * **Configurable Limits**: Control result count (1-100 chunks)
    * **Flexible Matching**: Choose optimal strategy based on query type

    Search Process:
    1. Parse request to determine search strategy and parameters
    2. Apply context filters to narrow search scope
    3. Execute search using appropriate algorithm (semantic/keyword/hybrid)
    4. Rank results by relevance score
    5. Optionally expand results with adjacent chunks for context

    Current Implementation Status:
    * ✅ Semantic Search: Fully implemented and production-ready
    * 🚧 Keyword Search: Planned feature, returns 400 if requested
    * 🚧 Hybrid Search: Planned feature, returns 400 if requested

    Notes:
    * Higher scores indicate better matches
    * Expansion increases response time but provides richer context
    * Use `/artifacts/list` to discover available collections and metadata
    * Semantic search works best for conceptual queries
    * Keyword search (when available) will excel at exact term matching
    """
    service: PrimitivesService = request.state.injector.get(PrimitivesService)
    response = service.search(body)
    return response
