from collections.abc import Iterator

from fastapi import APIRouter
from llama_index.llms import CompletionResponse
from starlette.responses import StreamingResponse

from private_gpt.di import root_injector
from private_gpt.open_ai.openai_models import OpenAICompletion
from private_gpt.query.query_service import QueryService

query_router = APIRouter()


@query_router.get("/query")
async def basic_completions(query: str) -> StreamingResponse:
    service = root_injector.get(QueryService)
    response_generator = service.stream_complete(query)
    return StreamingResponse(
        _to_openai_sse_stream(response_generator), media_type="text/event-stream"
    )


# TODO extract to a openai utils file
def _to_openai_sse_stream(
    response_generator: Iterator[CompletionResponse],
) -> Iterator[str]:
    for response in response_generator:
        yield f"data: {OpenAICompletion.simple_json_delta(text=response.delta)}\n\n"
    yield "data: [DONE]\n\n"
