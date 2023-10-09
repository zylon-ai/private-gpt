from dataclasses import dataclass

from fastapi import APIRouter
from llama_index.llms import ChatMessage, MessageRole
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from private_gpt.di import root_injector
from private_gpt.open_ai.extensions.context_docs import ContextDocs
from private_gpt.open_ai.openai_models import (
    OpenAICompletion,
    OpenAIMessage,
    to_openai_response,
    to_openai_sse_stream,
)
from private_gpt.server.chat.chat_service import ChatService

chat_router = APIRouter(prefix="/v1")


@dataclass
class ChatBody(BaseModel):
    messages: list[OpenAIMessage]
    context_docs: ContextDocs | None = None
    stream: bool | None = False


@chat_router.post("/chat/completions", response_model=None)
def chat_completion(body: ChatBody) -> OpenAICompletion | StreamingResponse:
    service = root_injector.get(ChatService)
    all_messages = [
        ChatMessage(content=m.content, role=MessageRole(m.role)) for m in body.messages
    ]
    if body.stream:
        stream = service.stream_chat(all_messages, body.context_docs)
        return StreamingResponse(
            to_openai_sse_stream(stream), media_type="text/event-stream"
        )
    else:
        response = service.chat(all_messages, body.context_docs)
        return to_openai_response(response)
