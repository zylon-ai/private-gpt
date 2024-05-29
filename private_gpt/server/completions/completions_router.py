from private_gpt.users import crud, models, schemas
import itertools
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from fastapi import APIRouter, Depends, Request, Security, HTTPException, status
from private_gpt.server.ingest.ingest_service import IngestService
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
import traceback
import logging
import json
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
import uuid
completions_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class CompletionsBody(BaseModel):
    conversation_id: uuid.UUID
    history: Optional[list[OpenAIMessage]]
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
                    "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "history": [
                        {
                            "role": "user",
                            "content": "Hello!"
                        },
                        {
                            "role": "assistant",
                            "content": "Hello, how can I help you?"
                        }
                    ],
                    "prompt": "How do you fry an egg?",
                    "system_prompt": "You are a rapper. Always answer with a rap.",
                    "stream": False,
                    "use_context": False,
                    "include_sources": False,
                }
            ]
        }
    }

class ChatContentCreate(BaseModel):
    content: Dict[str, Any]

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

def create_chat_item(db, sender, content, conversation_id):
    chat_item_create = schemas.ChatItemCreate(
            sender=sender,
            content=content,
            conversation_id=conversation_id
        )
    chat_history = crud.chat.get_conversation(db, conversation_id=conversation_id)
    chat_history.generate_title()
    return crud.chat_item.create(db, obj_in=chat_item_create)

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
        docs_ids = []
        for filename in docs_list:
            doc_id = service.get_doc_ids_by_filename(filename)
            docs_ids.extend(doc_id)
        body.context_filter = {"docs_ids": docs_ids}

        chat_history = crud.chat.get_by_id(
            db, id=body.conversation_id
        )
        if (chat_history is None) and (chat_history.user_id != current_user.id):
            raise HTTPException(
                status_code=404, detail="Chat not found")
        
        _history = body.history if body.history else []

        def build_history() -> list[OpenAIMessage]:
            history_messages: list[OpenAIMessage] = []
            for interaction in _history:
                role = interaction.role
                if role == 'user':
                    history_messages.append(
                        OpenAIMessage(
                            content=interaction.content,
                            role="user"
                        )
                    )
                else:
                    history_messages.append(
                        OpenAIMessage(
                            content=interaction.content,
                            role="assistant"
                        )
                    )
            return history_messages
        message = body.prompt 
        # message = body.prompt + 'Only answer if there is answer in the provided documents'
        user_message = OpenAIMessage(content=message, role="user")        
        user_message_json = {
            'text': body.prompt,
        }
        create_chat_item(db, "user", user_message_json , body.conversation_id) # store every query in the db
        
        messages = [user_message]

        if body.system_prompt:
            messages.insert(0, OpenAIMessage(
                content=body.system_prompt, role="system"))
            
        all_messages = [*build_history(), user_message]
        chat_body = ChatBody(
            messages=all_messages,
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
        
        chat_response = await chat_completion(request, chat_body)
        ai_response = chat_response.model_dump(mode="json")
        create_chat_item(db, "assistant", ai_response, body.conversation_id)
        return chat_response
    
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"There was an error: {str(e)}")
        raise