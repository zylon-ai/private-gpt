from fastapi import APIRouter, Depends, Request, Security, HTTPException, status
from private_gpt.server.ingest.ingest_service import IngestService
from pydantic import BaseModel
from sqlalchemy.orm import Session
import traceback
import logging

logger = logging.getLogger(__name__)

from starlette.responses import StreamingResponse

from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.open_ai.openai_models import (
    OpenAICompletion,
    OpenAIMessage,
)
from private_gpt.server.chat.chat_router import ChatBody, chat_completion
from private_gpt.server.utils.auth import authenticated
from private_gpt.users.api import deps
from private_gpt.users import crud, models, schemas
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


# @completions_router.post(
#     "/completions",
#     response_model=None,
#     summary="Completion",
#     responses={200: {"model": OpenAICompletion}},
#     tags=["Contextual Completions"],
# )
# def prompt_completion(
#     request: Request, body: CompletionsBody
# ) -> OpenAICompletion | StreamingResponse:
#     """We recommend most users use our Chat completions API.

#     Given a prompt, the model will return one predicted completion.

#     Optionally include a `system_prompt` to influence the way the LLM answers.

#     If `use_context`
#     is set to `true`, the model will use context coming from the ingested documents
#     to create the response. The documents being used can be filtered using the
#     `context_filter` and passing the document IDs to be used. Ingested documents IDs
#     can be found using `/ingest/list` endpoint. If you want all ingested documents to
#     be used, remove `context_filter` altogether.

#     When using `'include_sources': true`, the API will return the source Chunks used
#     to create the response, which come from the context provided.

#     When using `'stream': true`, the API will return data chunks following [OpenAI's
#     streaming model](https://platform.openai.com/docs/api-reference/chat/streaming):
#     ```
#     {"id":"12345","object":"completion.chunk","created":1694268190,
#     "model":"private-gpt","choices":[{"index":0,"delta":{"content":"Hello"},
#     "finish_reason":null}]}
#     ```
#     """
#     messages = [OpenAIMessage(content=body.prompt, role="user")]
#     # If system prompt is passed, create a fake message with the system prompt.
#     if body.system_prompt:
#         messages.insert(0, OpenAIMessage(content=body.system_prompt, role="system"))

#     chat_body = ChatBody(
#         messages=messages,
#         use_context=body.use_context,
#         stream=body.stream,
#         include_sources=body.include_sources,
#         context_filter=body.context_filter,
#     )
#     return chat_completion(request, chat_body)


@completions_router.post(
    "/chat",
    response_model=None,
    summary="Completion",
    responses={200: {"model": OpenAICompletion}},
    tags=["Contextual Completions"],
    openapi_extra={
        "x-fern-streaming": {
            "stream-condition": "stream",
            "response": {"$ref": "#/components/schemas/OpenAICompletion"},
            "response-stream": {"$ref": "#/components/schemas/OpenAICompletion"},
        }
    },
)
async def prompt_completion(
    request: Request,
    body: CompletionsBody,
    db: Session = Depends(deps.get_db),
    log_audit: models.Audit = Depends(deps.get_audit_logger),
    current_user: models.User = Security(
        deps.get_current_user,
    ),
) -> OpenAICompletion | StreamingResponse:
    
    service = request.state.injector.get(IngestService)
    try:
        department = crud.department.get_by_id(
            db, id=current_user.department_id)
        if not department:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"No department assigned to you")
        documents = crud.documents.get_enabled_documents_by_departments(
            db, department_id=department.id)
        if not documents:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"No documents uploaded for your department.")
        docs_list = [document.filename for document in documents]
        print("DOCUMENTS ASSIGNED TO THIS DEPARTMENTS: ", docs_list)
        docs_ids = []
        for filename in docs_list:
            doc_id = service.get_doc_ids_by_filename(filename)
            docs_ids.extend(doc_id)
        body.context_filter = {"docs_ids": docs_ids}

    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"There was an error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )

    messages = [OpenAIMessage(content=body.prompt, role="user")]
    if body.system_prompt:
        messages.insert(0, OpenAIMessage(
            content=body.system_prompt, role="system"))

    chat_body = ChatBody(
        messages=messages,
        use_context=body.use_context,
        stream=body.stream,
        include_sources=body.include_sources,
        context_filter=body.context_filter,
    )
    log_audit(
        model='Chat', 
        action='Chat',
        details={
            "query": body.prompt,
            'user': current_user.username,
            }, 
        user_id=current_user.id
    )
    return await chat_completion(request, chat_body)
