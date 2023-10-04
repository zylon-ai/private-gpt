from dataclasses import dataclass

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from private_gpt.di import root_injector
from private_gpt.llm.llm_service import LLMService
from private_gpt.open_ai.openai_models import to_openai_sse_stream

completions_router = APIRouter()


@dataclass
class CompletionsBody(BaseModel):
    prompt: str


@completions_router.post("/completions")
def completions(body: CompletionsBody) -> StreamingResponse:
    return _run_llm(body.prompt)


@completions_router.get("/completions")
def basic_completions(prompt: str) -> StreamingResponse:
    return _run_llm(prompt)


def _run_llm(prompt: str) -> StreamingResponse:
    service = root_injector.get(LLMService)
    stream = service.stream_complete(prompt)
    return StreamingResponse(
        to_openai_sse_stream(stream), media_type="text/event-stream"
    )
