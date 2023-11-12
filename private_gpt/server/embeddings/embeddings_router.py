from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from private_gpt.di import root_injector
from private_gpt.server.embeddings.embeddings_service import (
    Embedding,
    EmbeddingsService,
)
from private_gpt.server.utils.auth import authenticated

embeddings_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class EmbeddingsBody(BaseModel):
    input: str | list[str]


class EmbeddingsResponse(BaseModel):
    object: Literal["list"]
    model: Literal["private-gpt"]
    data: list[Embedding]


@embeddings_router.post("/embeddings", tags=["Embeddings"])
def embeddings_generation(body: EmbeddingsBody) -> EmbeddingsResponse:
    """Get a vector representation of a given input.

    That vector representation can be easily consumed
    by machine learning models and algorithms.
    """
    service = root_injector.get(EmbeddingsService)
    input_texts = body.input if isinstance(body.input, list) else [body.input]
    embeddings = service.texts_embeddings(input_texts)
    return EmbeddingsResponse(object="list", model="private-gpt", data=embeddings)
