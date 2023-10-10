from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.open_ai.openai_models import (
    OpenAICompletion,
    OpenAIMessage,
)
from private_gpt.server.chat.chat_router import ChatBody, chat_completion

completions_router = APIRouter(prefix="/v1")


class CompletionsBody(BaseModel):
    prompt: str
    use_context: bool = False
    context_filter: ContextFilter | None = None
    stream: bool = False


@completions_router.post("/completions", response_model=None)
def prompt_completion(body: CompletionsBody) -> OpenAICompletion | StreamingResponse:
    """Deprecated. Use /chat/completions instead.

    This endpoint only exists for openai compatibility.
    """
    message = OpenAIMessage(content=body.prompt, role="user")
    chat_body = ChatBody(
        messages=[message],
        use_context=body.use_context,
        stream=body.stream,
        context_filter=body.context_filter,
    )
    return chat_completion(chat_body)
