from dataclasses import dataclass

from fastapi import APIRouter
from pydantic import BaseModel

from private_gpt.di import root_injector
from private_gpt.server.embeddings.embeddings_service import EmbeddingsService

embeddings_router = APIRouter(prefix="/v1")


@dataclass
class EmbeddingsBody(BaseModel):
    input: str | list[str]


@dataclass
class Embedding:
    index: int
    object: str
    embedding: list[float]


@dataclass
class EmbeddingsResponse:
    object: str
    model: str
    data: list[Embedding]


@embeddings_router.post("/embeddings")
def embeddings_generation(body: EmbeddingsBody) -> EmbeddingsResponse:
    service = root_injector.get(EmbeddingsService)
    input_texts = body.input if isinstance(body.input, list) else [body.input]
    embeddings = service.texts_embeddings(input_texts)
    embeddings_result = [
        Embedding(
            index=embeddings.index(embedding), object="embedding", embedding=embedding
        )
        for embedding in embeddings
    ]
    return EmbeddingsResponse(
        object="list", model="private-gpt", data=embeddings_result
    )
