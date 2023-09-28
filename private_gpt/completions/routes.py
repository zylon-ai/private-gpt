from collections.abc import Iterator

from fastapi import APIRouter
from llama_index.llms import CompletionResponse
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from private_gpt.completions.completions_service import CompletionsService
from private_gpt.di import root_injector
from private_gpt.dto.openai import OpenAICompletion

completions_router = APIRouter()


class CompletionsBody(BaseModel):
    prompt: str
    model: str | None = None


@completions_router.post("/completions")
async def completions(body: CompletionsBody) -> StreamingResponse:
    return _run_llm(body)


@completions_router.get("/completions")
async def basic_completions(prompt: str, model: str | None = None) -> StreamingResponse:
    body = CompletionsBody(prompt=prompt, model=model)
    return _run_llm(body)


def _run_llm(body: CompletionsBody) -> StreamingResponse:
    service = root_injector.get(CompletionsService)
    response_generator = service.stream_complete(body.prompt, model=body.model)
    return StreamingResponse(
        _to_openai_sse_stream(response_generator), media_type="text/event-stream"
    )


def _to_openai_sse_stream(
    response_generator: Iterator[CompletionResponse],
) -> Iterator[str]:
    for response in response_generator:
        yield f"data: {OpenAICompletion.simple_json_delta(text=response.delta)}\n\n"
    yield "data: [DONE]\n\n"
