import uuid
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import desc, asc

from private_gpt.users.crud.base import CRUDBase
from private_gpt.users.models.chat import ChatHistory, ChatItem
from private_gpt.users.schemas.chat import ChatHistoryCreate, ChatHistoryCreate, ChatItemCreate, ChatItemUpdate


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
        
    def get_conversation(self, db: Session, conversation_id: uuid.UUID) -> Optional[ChatHistory]:
         return (
                db.query(self.model)
                .filter(ChatHistory.conversation_id == conversation_id)
                .first()
            )
    
    def get_chat_history(
            self, db: Session, *,user_id:int
        ) -> List[ChatHistory]:
            return (
                db.query(self.model)
                .filter(ChatHistory.user_id == user_id)
                .order_by(desc(getattr(ChatHistory, 'created_at')))
                .all()
            )
    
class CRUDChatItem(CRUDBase[ChatItem, ChatItemCreate, ChatItemUpdate]):
    pass


chat = CRUDChat(ChatHistory)
chat_item = CRUDChatItem(ChatItem)
