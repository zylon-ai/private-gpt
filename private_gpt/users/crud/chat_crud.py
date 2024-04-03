from typing import Optional, List, Union, Dict, Any
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from private_gpt.users.crud.base import CRUDBase
from private_gpt.users.models.chat import ChatHistory, ChatItem
from private_gpt.users.schemas.chat import ChatHistoryCreate, ChatHistoryCreate, ChatItemCreate, ChatItemUpdate


class CRUDChat(CRUDBase[ChatHistory, ChatHistoryCreate, ChatHistoryCreate]):
    def get_by_id(self, db: Session, *, id: int) -> Optional[ChatHistory]:
        return db.query(self.model).filter(ChatHistory.conversation_id == id).first()


class CRUDChatItem(CRUDBase[ChatItem, ChatItemCreate, ChatItemUpdate]):
    pass


chat = CRUDChat(ChatHistory)
chat_item = CRUDChatItem(ChatItem)
