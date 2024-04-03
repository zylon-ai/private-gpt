from typing import Optional, List, Union, Dict, Any
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.util import object_mapper

from private_gpt.users.crud.base import CRUDBase
from private_gpt.users.models.chat_history import ChatHistory
from private_gpt.users.schemas.chat_history import ChatCreate, ChatUpdate


class CRUDChat(CRUDBase[ChatHistory, ChatCreate, ChatUpdate]):
    def get_by_id(self, db: Session, *, id: int) -> Optional[ChatHistory]:
        return db.query(self.model).filter(ChatHistory.conversation_id == id).first()

    def update_messages(
        self,
        db: Session,
        *,
        db_obj: ChatHistory,
        obj_in: Union[ChatUpdate, Dict[str, Any]]
    ) -> ChatHistory:
        try:
            obj_data = object_mapper(db_obj).data
            if isinstance(obj_in, dict):
                update_data = obj_in
            else:
                update_data = obj_in.dict(exclude_unset=True)

            # Update the `messages` field by appending new messages
            existing_messages = obj_data.get("messages", [])
            new_messages = update_data.get("messages", [])
            obj_data["messages"] = existing_messages + new_messages

            for field, value in obj_data.items():
                setattr(db_obj, field, value)

            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            return db_obj
        except IntegrityError as e:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Integrity Error: {str(e)}",
            )


chat = CRUDChat(ChatHistory)
