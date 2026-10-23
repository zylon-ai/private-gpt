from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from private_gpt.server.embeddings.embeddings_service import (
    Embedding,
    EmbeddingsService,
)
from private_gpt.server.utils.auth import authenticated
from private_gpt.server.utils.openapi_models import OpenAPIValidationErrorResponse

embeddings_router = APIRouter(
    prefix="/v1",
    dependencies=[Depends(authenticated)],
    tags=["Embeddings"],
    responses={401: {"description": "Unauthorized"}},
)


class EmbeddingsBody(BaseModel):
    """Request body for generating vector embeddings from text input."""

    model: str = Field(default="default", description="Model identifier or alias.")
    input: str | list[str] = Field(
        ...,
        title="EmbeddingsInput",
        description=(
            "The text(s) to generate embeddings for. Can be a single string "
            "or an array of strings. Each text should be a meaningful unit "
            "of text (sentence, paragraph, etc)."
        ),
        examples=["The quick brown fox jumps over the lazy dog"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"input": "The quick brown fox jumps over the lazy dog"},
                {
                    "model": "custom-embedding-model",
                    "input": [
                        "Machine learning is fascinating",
                        "Neural networks process data efficiently",
                        "Embeddings represent text as vectors",
                    ],
                },
            ]
        }
    }


class EmbeddingsResponse(BaseModel):
    """Response containing generated embeddings for input text(s)."""

    object: Literal["list"] = Field(
        default="list", description="The type of object returned"
    )
    model: Literal["private-gpt"] = Field(
        default="private-gpt", description="The model used to generate embeddings"
    )
    data: list[Embedding] = Field(
        ..., description="List of embeddings, one for each input text"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "object": "list",
                    "model": "private-gpt",
                    "data": [
                        {
                            "index": 0,
                            "object": "embedding",
                            "embedding": [0.123, -0.456, 0.789, 0.234, -0.567, 0.891],
                        },
                    ],
                },
                {
                    "object": "list",
                    "model": "private-gpt",
                    "data": [
                        {
                            "index": 0,
                            "object": "embedding",
                            "embedding": [0.234, -0.567, 0.123],
                        },
                        {
                            "index": 1,
                            "object": "embedding",
                            "embedding": [-0.123, 0.456, -0.789],
                        },
                    ],
                },
            ]
        }
    }


@embeddings_router.post(
    "/embeddings",
    response_model=EmbeddingsResponse,
    summary="Generate embeddings for text input",
    responses={
        200: {
            "description": "Successfully generated embeddings",
            "content": {
                "application/json": {
                    "examples": {
                        "single_text_embedding": {
                            "summary": "Embedding for single text input",
                            "value": {
                                "object": "list",
                                "model": "private-gpt",
                                "data": [
                                    {
                                        "index": 0,
                                        "object": "embedding",
                                        "embedding": [
                                            0.123,
                                            -0.456,
                                            0.789,
                                            0.234,
                                            -0.567,
                                            0.891,
                                            -0.123,
                                            0.445,
                                            -0.678,
                                            0.901,
                                        ],
                                    }
                                ],
                            },
                        },
                        "multiple_text_embeddings": {
                            "summary": "Embeddings for multiple text inputs",
                            "value": {
                                "object": "list",
                                "model": "private-gpt",
                                "data": [
                                    {
                                        "index": 0,
                                        "object": "embedding",
                                        "embedding": [
                                            0.234,
                                            -0.567,
                                            0.123,
                                            0.678,
                                            -0.234,
                                            0.456,
                                            -0.789,
                                            0.345,
                                            -0.123,
                                            0.567,
                                        ],
                                    },
                                    {
                                        "index": 1,
                                        "object": "embedding",
                                        "embedding": [
                                            -0.123,
                                            0.456,
                                            -0.789,
                                            0.234,
                                            0.567,
                                            -0.345,
                                            0.678,
                                            -0.234,
                                            0.789,
                                            -0.456,
                                        ],
                                    },
                                    {
                                        "index": 2,
                                        "object": "embedding",
                                        "embedding": [
                                            0.567,
                                            -0.234,
                                            0.789,
                                            -0.456,
                                            0.123,
                                            0.678,
                                            -0.345,
                                            0.234,
                                            -0.567,
                                            0.891,
                                        ],
                                    },
                                ],
                            },
                        },
                        "semantic_similarity_example": {
                            "summary": "Embeddings for semantic similarity comparison",
                            "value": {
                                "object": "list",
                                "model": "private-gpt",
                                "data": [
                                    {
                                        "index": 0,
                                        "object": "embedding",
                                        "embedding": [
                                            0.445,
                                            0.223,
                                            -0.667,
                                            0.112,
                                            0.889,
                                            -0.334,
                                            0.556,
                                            -0.778,
                                            0.223,
                                            0.445,
                                        ],
                                    },
                                    {
                                        "index": 1,
                                        "object": "embedding",
                                        "embedding": [
                                            0.434,
                                            0.234,
                                            -0.656,
                                            0.123,
                                            0.878,
                                            -0.345,
                                            0.567,
                                            -0.789,
                                            0.234,
                                            0.456,
                                        ],
                                    },
                                ],
                            },
                        },
                    }
                }
            },
        },
        422: {
            "model": OpenAPIValidationErrorResponse,
            "description": "Validation Error - Invalid input format",
            "content": {
                "application/json": {
                    "examples": {
                        "missing_input": {
                            "summary": "Missing input field",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "input"],
                                        "msg": "field required",
                                        "type": "value_error.missing",
                                    }
                                ]
                            },
                        },
                        "empty_input": {
                            "summary": "Empty input provided",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "input"],
                                        "msg": "Input text cannot be empty",
                                        "type": "value_error",
                                    }
                                ]
                            },
                        },
                        "invalid_input_type": {
                            "summary": "Invalid input data type",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "input"],
                                        "msg": "Input must be a string or array of strings",
                                        "type": "type_error",
                                    }
                                ]
                            },
                        },
                        "empty_array": {
                            "summary": "Empty array provided",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "input"],
                                        "msg": "Input array cannot be empty",
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
    tags=["Embeddings"],
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/EmbeddingsBody"}
                }
            },
            "required": True,
            "description": (
                "Request body for generating vector embeddings from text.\n\n"
                "Contains input text(s) to be converted into high-dimensional "
                "vector representations. Supports both single string input "
                "and batch processing with arrays of strings for efficient "
                "embedding generation.\n\n"
                "The request body defines text input and processing options "
                "for generating semantic embeddings that capture meaning "
                "and relationships between texts."
            ),
        }
    },
)
def embeddings_generation(request: Request, body: EmbeddingsBody) -> EmbeddingsResponse:
    """Generate vector embeddings from input text.

    This endpoint converts text into high-dimensional vector representations
    that capture semantic meaning. These embeddings preserve semantic
    relationships between texts and can be used for various machine learning
    tasks.

    Notes:
    * Empty strings or arrays are not accepted
    * Results include index numbers for mapping back to original inputs
    * All embeddings are generated as query vectors. If the embeddings have two modes
    (e.g., query and document), this endpoint only returns query vectors.
    """
    service = request.state.injector.get(EmbeddingsService)
    embeddings = service.texts_embeddings(
        model=body.model,
        texts=body.input if isinstance(body.input, list) else [body.input],
    )
    return EmbeddingsResponse(object="list", model="private-gpt", data=embeddings)
