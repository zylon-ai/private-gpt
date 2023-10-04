from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from private_gpt.di import root_injector
from private_gpt.open_ai.openai_models import to_openai_sse_stream
from private_gpt.query.query_service import QueryService

query_router = APIRouter()


@query_router.get("/query")
def contextualized_query(query: str) -> StreamingResponse:
    service = root_injector.get(QueryService)
    stream = service.stream_complete(query)
    return StreamingResponse(
        to_openai_sse_stream(stream), media_type="text/event-stream"
    )
