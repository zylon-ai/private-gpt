from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.utils.auth import authenticated

chunks_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class ChunksBody(BaseModel):
    text: str = Field(examples=["Q3 2023 sales"])
    context_filter: ContextFilter | None = None
    limit: int = 10
    prev_next_chunks: int = Field(default=0, examples=[2])


class ChunksResponse(BaseModel):
    object: Literal["list"]
    model: Literal["private-gpt"]
    data: list[Chunk]


@chunks_router.post("/chunks", tags=["Context Chunks"])
def chunks_retrieval(request: Request, body: ChunksBody) -> ChunksResponse:
    """Given a `text`, returns the most relevant chunks from the ingested documents.

    The returned information can be used to generate prompts that can be
    passed to `/completions` or `/chat/completions` APIs. Note: it is usually a very
    fast API, because only the Embeddings model is involved, not the LLM. The
    returned information contains the relevant chunk `text` together with the source
    `document` it is coming from. It also contains a score that can be used to
    compare different results.

    The max number of chunks to be returned is set using the `limit` param.

    Previous and next chunks (pieces of text that appear right before or after in the
    document) can be fetched by using the `prev_next_chunks` field.

    The documents being used can be filtered using the `context_filter` and passing
    the document IDs to be used. Ingested documents IDs can be found using
    `/ingest/list` endpoint. If you want all ingested documents to be used,
    remove `context_filter` altogether.
    """
    service = request.state.injector.get(ChunksService)
    results = service.retrieve_relevant(
        body.text, body.context_filter, body.limit, body.prev_next_chunks
    )
    return ChunksResponse(
        object="list",
        model="private-gpt",
        data=results,
    )
