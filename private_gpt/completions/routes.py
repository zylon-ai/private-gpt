from collections.abc import AsyncIterator

from fastapi import APIRouter
from llama_index.llms import ChatMessage, ChatResponse, CompletionResponse
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from private_gpt.completions.completions_service import CompletionsService
from private_gpt.di import root_injector
from private_gpt.open_ai.openai_models import OpenAICompletion

completions_router = APIRouter()


class CompletionsBody(BaseModel):
    prompt: str


@completions_router.post("/completions")
async def completions(body: CompletionsBody) -> StreamingResponse:
    return await _run_llm(body.prompt)


@completions_router.get("/completions")
async def basic_completions(prompt: str) -> StreamingResponse:
    return await _run_llm(prompt)


async def _run_llm(prompt: str) -> StreamingResponse:
    service = root_injector.get(CompletionsService)
    stream = await service.stream_complete(prompt)
    return StreamingResponse(
        _to_openai_sse_stream(stream), media_type="text/event-stream"
    )


async def _to_openai_sse_stream(
    response_generator: AsyncIterator[CompletionResponse | ChatResponse],
) -> AsyncIterator[str]:
    async for response in response_generator:
        yield f"data: {OpenAICompletion.simple_json_delta(text=response.delta)}\n\n"
    yield "data: [DONE]\n\n"
