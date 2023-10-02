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
    model: str | None = None


@completions_router.post("/completions")
async def completions(body: CompletionsBody) -> StreamingResponse:
    return await _run_llm(body)


@completions_router.get("/completions")
async def basic_completions(prompt: str, model: str | None = None) -> StreamingResponse:
    body = CompletionsBody(prompt=prompt, model=model)
    return await _run_llm(body)


async def _run_llm(body: CompletionsBody) -> StreamingResponse:
    service = root_injector.get(CompletionsService)
    message = ChatMessage(content=body.prompt)
    stream = await service.stream_chat([message], model_name=body.model)
    return StreamingResponse(
        _to_openai_sse_stream(stream), media_type="text/event-stream"
    )


async def _to_openai_sse_stream(
    response_generator: AsyncIterator[CompletionResponse | ChatResponse],
) -> AsyncIterator[str]:
    async for response in response_generator:
        yield f"data: {OpenAICompletion.simple_json_delta(text=response.delta)}\n\n"
    yield "data: [DONE]\n\n"
