from fastapi import APIRouter, Depends, Request

from private_gpt.chat.input_models import CompletionInput, CompletionOutput
from private_gpt.server.chat.chat_request_mapper import ChatRequestMapper
from private_gpt.server.chat.chat_service import ChatService
from private_gpt.server.completion.completion_service import CompletionService
from private_gpt.server.utils.auth import authenticated

completion_router = APIRouter(
    prefix="/v1",
    dependencies=[Depends(authenticated)],
    tags=["Completions"],
    responses={401: {"description": "Unauthorized"}},
)


@completion_router.post(
    "/complete",
    response_model=CompletionOutput,
    summary="Create a Text Completion",
    tags=["Completions"],
)
async def create_completion(
    request: Request,
    body: CompletionInput,
) -> CompletionOutput:
    chat_service: ChatService = request.state.injector.get(ChatService)
    request_mapper: ChatRequestMapper = request.state.injector.get(ChatRequestMapper)
    completion_service: CompletionService = request.state.injector.get(
        CompletionService
    )

    chat_body = completion_service.to_chat_body(body)
    chat_request = await request_mapper.create_request_from_body(chat_body)

    completion = await chat_service.chat(chat_request)
    completion_text = completion_service.extract_text_from_content(completion.content)

    return completion_service.to_completion_output(
        completion=completion_text,
        stop_reason=completion.stop_reason,
        model=body.model,
    )
