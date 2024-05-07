import logging
import traceback
import uuid
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Depends, HTTPException, status, Security
from fastapi_pagination import Page, paginate

from private_gpt.users.api import deps
from private_gpt.users import crud, models, schemas

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/c", tags=["Chat Histories"])


@router.get("", response_model=Page[schemas.Chat])
def list_chat_histories(
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
    ),
) -> Page[schemas.Chat]:
    """
    Retrieve a list of chat histories with pagination support.
    """
    try:
        chat_histories = crud.chat.get_chat_history(
            db, user_id=current_user.id)
        return paginate(chat_histories)
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Error listing chat histories: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error",
        )


@router.post("/create", response_model=schemas.ChatHistory)
def create_chat_history(
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
    ),
) -> schemas.ChatHistory:
    """
    Create a new chat history
    """
    try:
        chat_history_in = schemas.CreateChatHistory(
            user_id= current_user.id
        )
        chat_history = crud.chat.create(
            db=db, obj_in=chat_history_in)
        return chat_history
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Error creating chat history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error",
        )


@router.get("/{conversation_id}", response_model=schemas.ChatHistory)
def read_chat_history(
    conversation_id: uuid.UUID,
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
    ),
) -> schemas.ChatHistory:
    """
    Read a chat history by ID
    """
    try:
        chat_history = crud.chat.get_by_id(db, id=conversation_id, skip=skip, limit=limit)
        if chat_history is None or chat_history.user_id != current_user.id:
            raise HTTPException(
                status_code=404, detail="Chat history not found")
        return chat_history
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Error reading chat history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error",
        )


@router.post("/delete")
def delete_chat_history(
    chat_history_in: schemas.ChatDelete,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
    ),
):
    """
    Delete a chat history by ID
    """
    try:
        chat_history_id = chat_history_in.conversation_id
        chat_history = crud.chat.get_by_id(db, id=chat_history_id)
        if chat_history is None or chat_history.user_id != current_user.id:
            raise HTTPException(
                status_code=404, detail="Chat history not found")

        crud.chat.remove(db=db, id=chat_history_id)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Chat history deleted successfully",
            },
        )
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Error deleting chat history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error",
        )


