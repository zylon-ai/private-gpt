from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.open_ai.openai_models import (
    OpenAICompletion,
    OpenAIMessage,
)
from private_gpt.server.chat.chat_router import ChatBody, chat_completion
from private_gpt.server.utils.auth import authenticated

completions_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class CompletionsBody(BaseModel):
    prompt: str
    system_prompt: str | None = None
    use_context: bool = False
    context_filter: ContextFilter | None = None
    include_sources: bool = True
    stream: bool = False

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "How do you fry an egg?",
                    "system_prompt": "You are a rapper. Always answer with a rap.",
                    "stream": False,
                    "use_context": False,
                    "include_sources": False,
                }
            ]
        }
    }


@completions_router.post(
    "/completions",
    response_model=None,
    summary="Completion",
    responses={200: {"model": OpenAICompletion}},
    tags=["Contextual Completions"],
)
def prompt_completion(
    request: Request, body: CompletionsBody
) -> OpenAICompletion | StreamingResponse:
    """We recommend most users use our Chat completions API.

    Given a prompt, the model will return one predicted completion.

    Optionally include a `system_prompt` to influence the way the LLM answers.

    If `use_context`
    is set to `true`, the model will use context coming from the ingested documents
    to create the response. The documents being used can be filtered using the
    `context_filter` and passing the document IDs to be used. Ingested documents IDs
    can be found using `/ingest/list` endpoint. If you want all ingested documents to
    be used, remove `context_filter` altogether.

    When using `'include_sources': true`, the API will return the source Chunks used
    to create the response, which come from the context provided.

    When using `'stream': true`, the API will return data chunks following [OpenAI's
    streaming model](https://platform.openai.com/docs/api-reference/chat/streaming):
    ```
    {"id":"12345","object":"completion.chunk","created":1694268190,
    "model":"private-gpt","choices":[{"index":0,"delta":{"content":"Hello"},
    "finish_reason":null}]}
    ```
    """
    messages = [OpenAIMessage(content=body.prompt, role="user")]
    # If system prompt is passed, create a fake message with the system prompt.
    if body.system_prompt:
        messages.insert(0, OpenAIMessage(content=body.system_prompt, role="system"))

    chat_body = ChatBody(
        messages=messages,
        use_context=body.use_context,
        stream=body.stream,
        include_sources=body.include_sources,
        context_filter=body.context_filter,
    )
    return chat_completion(request, chat_body)
