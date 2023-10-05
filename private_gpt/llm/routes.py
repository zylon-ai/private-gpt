from dataclasses import dataclass

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
from llama_index.llms import ChatMessage, MessageRole
from pydantic import BaseModel

from private_gpt.di import root_injector
from private_gpt.llm.llm_service import LLMService
from private_gpt.open_ai.openai_models import (
    OpenAIMessage,
    to_openai_sse_stream,
    to_openai_response,
    OpenAICompletion,
)

completions_router = APIRouter(prefix="/v1")


@dataclass
class CompletionsBody(BaseModel):
    prompt: str


@dataclass
class ChatCompletionBody(BaseModel):
    messages: list[OpenAIMessage]
    stream: bool = False


@completions_router.post("/completions")
def completions(body: CompletionsBody) -> StreamingResponse:
    return _run_llm(body.prompt)


@completions_router.get("/completions")
def basic_completions(prompt: str) -> StreamingResponse:
    return _run_llm(prompt)


@completions_router.post("/chat/completions", response_model=None)
def chat_completions(body: ChatCompletionBody) -> OpenAICompletion | StreamingResponse:
    return _run_chat(body)


def _run_chat(body: ChatCompletionBody) -> JSONResponse | StreamingResponse:
    service = root_injector.get(LLMService)
    all_messages = [
        ChatMessage(content=m.content, role=MessageRole(m.role)) for m in body.messages
    ]
    if body.stream:
        stream = service.stream_chat(all_messages)
        return StreamingResponse(
            to_openai_sse_stream(stream), media_type="text/event-stream"
        )
    else:
        response = service.chat(all_messages)
        openai_response = to_openai_response(response)
        return JSONResponse(content=jsonable_encoder(openai_response))


def _run_llm(prompt: str) -> StreamingResponse:
    service = root_injector.get(LLMService)
    stream = service.stream_complete(prompt)
    return StreamingResponse(
        to_openai_sse_stream(stream), media_type="text/event-stream"
    )
