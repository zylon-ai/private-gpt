from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.open_ai.openai_models import (
    to_openai_sse_stream,
)
from private_gpt.server.recipes.summarize.summarize_service import SummarizeService
from private_gpt.server.utils.auth import authenticated

summarize_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class SummarizeBody(BaseModel):
    text: str | None = None
    use_context: bool = False
    context_filter: ContextFilter | None = None
    prompt: str | None = None
    instructions: str | None = None
    stream: bool = False


class SummarizeResponse(BaseModel):
    summary: str


@summarize_router.post(
    "/summarize",
    response_model=None,
    summary="Summarize",
    responses={200: {"model": SummarizeResponse}},
    tags=["Recipes"],
)
def summarize(
    request: Request, body: SummarizeBody
) -> SummarizeResponse | StreamingResponse:
    """Given a text, the model will return a summary.

    Optionally include `instructions` to influence the way the summary is generated.

    If `use_context`
    is set to `true`, the model will also use the content coming from the ingested
    documents in the summary. The documents being used can
    be filtered by their metadata using the `context_filter`.
    Ingested documents metadata can be found using `/ingest/list` endpoint.
    If you want all ingested documents to be used, remove `context_filter` altogether.

    If `prompt` is set, it will be used as the prompt for the summarization,
    otherwise the default prompt will be used.

    When using `'stream': true`, the API will return data chunks following [OpenAI's
    streaming model](https://platform.openai.com/docs/api-reference/chat/streaming):
    ```
    {"id":"12345","object":"completion.chunk","created":1694268190,
    "model":"private-gpt","choices":[{"index":0,"delta":{"content":"Hello"},
    "finish_reason":null}]}
    ```
    """
    service: SummarizeService = request.state.injector.get(SummarizeService)

    if body.stream:
        completion_gen = service.stream_summarize(
            text=body.text,
            instructions=body.instructions,
            use_context=body.use_context,
            context_filter=body.context_filter,
            prompt=body.prompt,
        )
        return StreamingResponse(
            to_openai_sse_stream(
                response_generator=completion_gen,
            ),
            media_type="text/event-stream",
        )
    else:
        completion = service.summarize(
            text=body.text,
            instructions=body.instructions,
            use_context=body.use_context,
            context_filter=body.context_filter,
            prompt=body.prompt,
        )
        return SummarizeResponse(
            summary=completion,
        )
