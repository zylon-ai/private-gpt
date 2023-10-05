from dataclasses import dataclass

from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from private_gpt.di import root_injector
from private_gpt.open_ai.extensions.context_files import ContextFiles
from private_gpt.open_ai.openai_models import (
    OpenAICompletion,
    to_openai_response,
    to_openai_sse_stream,
)
from private_gpt.server.completions.completions_service import CompletionsService

completions_router = APIRouter(prefix="/v1")


@dataclass
class CompletionsBody(BaseModel):
    prompt: str
    context_files: ContextFiles | None = None
    stream: bool | None = False


@completions_router.post("/completions", response_model=None)
def prompt_completion(body: CompletionsBody) -> OpenAICompletion | StreamingResponse:
    service = root_injector.get(CompletionsService)
    if body.stream:
        stream = service.stream_complete(body.prompt, body.context_files)
        return StreamingResponse(
            to_openai_sse_stream(stream), media_type="text/event-stream"
        )
    else:
        response = service.complete(body.prompt, body.context_files)
        return to_openai_response(response)
