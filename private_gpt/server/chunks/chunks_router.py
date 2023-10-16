import time
import uuid

from fastapi import APIRouter
from pydantic import BaseModel

from private_gpt.di import root_injector
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService

chunks_router = APIRouter(prefix="/v1")


class ChunksBody(BaseModel):
    text: str
    context_filter: ContextFilter | None = None
    limit: int = 10
    prev_next_chunks: int = 0


class ChunksResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    chunks: list[Chunk]


@chunks_router.post("/chunks", tags=["Chunks"])
def chunks_retrieval(body: ChunksBody) -> ChunksResponse:
    service = root_injector.get(ChunksService)
    results = service.retrieve_relevant(
        body.text, body.context_filter, body.limit, body.prev_next_chunks
    )
    return ChunksResponse(
        id=str(uuid.uuid4()),
        object="file.chunk",
        created=int(time.time()),
        model="private-gpt",
        chunks=results,
    )
