from sqlalchemy.sql.expression import desc, asc
from typing import Optional, List, Union, Dict, Any
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import uuid
from private_gpt.users.crud.base import CRUDBase
from private_gpt.users.models.chat import ChatHistory, ChatItem
from private_gpt.users.schemas.chat import ChatHistoryCreate, ChatHistoryCreate, ChatItemCreate, ChatItemUpdate
from fastapi_pagination import Page, paginate


class CRUDChat(CRUDBase[ChatHistory, ChatHistoryCreate, ChatHistoryCreate]):
    def get_by_id(self, db: Session, *, id: uuid.UUID, skip: int=0, limit: int=10) -> Optional[ChatHistory]:
        chat_history = (
            db.query(self.model)
            .filter(ChatHistory.conversation_id == id)
            .order_by(asc(getattr(ChatHistory, 'created_at')))
            .first()
        )
        if chat_history:
            chat_history.chat_items = (
                db.query(ChatItem)
                .filter(ChatItem.conversation_id == id)
                .order_by(desc(getattr(ChatItem, 'index')))
                .offset(skip)
                .limit(limit)
                .all()
            )
        return chat_history

    def get_chat_history(
            self, db: Session, *,user_id:int, skip: int = 0, limit: int =100
        ) -> List[ChatHistory]:
            return (
                db.query(self.model)
                .filter(ChatHistory.user_id == user_id)
                .order_by(desc(getattr(ChatHistory, 'created_at')))
                .offset(skip)
                .limit(limit)
                .all()
            )
    
class CRUDChatItem(CRUDBase[ChatItem, ChatItemCreate, ChatItemUpdate]):
    pass


chat = CRUDChat(ChatHistory)
chat_item = CRUDChatItem(ChatItem)
