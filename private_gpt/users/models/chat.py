import uuid
from datetime import datetime
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID 
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, event, JSON

from private_gpt.users.db.base_class import Base

class ChatHistory(Base):
    """Models a chat history table"""

    __tablename__ = "chat_history"

    conversation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now,
                        onupdate=datetime.now)
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="chat_histories")
    chat_items = relationship(
        "ChatItem", back_populates="chat_history", cascade="all, delete-orphan")
    _title_generated = Column(Boolean, default=False)

    def __init__(self, user_id, chat_items=None, **kwargs):
        super().__init__(**kwargs)
        self.user_id = user_id
        self.chat_items = chat_items or []
        self.generate_title()

    def generate_title(self):
        user_chat_items = [
            item for item in self.chat_items if item.sender == "user"]
        if user_chat_items:
            first_user_chat_item = user_chat_items[0]
            self.title = first_user_chat_item.content[:30]
        else:
            self.title = str(self.conversation_id)

    def __repr__(self):
        """Returns string representation of model instance"""
        return f"<ChatHistory {self.conversation_id!r}>"


class ChatItem(Base):
    """Models a chat item table"""

    __tablename__ = "chat_items"

    id = Column(Integer, nullable=False, primary_key=True)
    sender = Column(String(225), nullable=False)
    content = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now,
                        onupdate=datetime.now)
    like = Column(Boolean, default=True)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey(
        "chat_history.conversation_id"), nullable=False)
    chat_history = relationship("ChatHistory", back_populates="chat_items")

    def __repr__(self):
        """Returns string representation of model instance"""
        return f"<ChatItem {self.id!r}>"


@event.listens_for(ChatHistory, "after_insert")
def receive_after_insert(mapper, connection, target):
    """Update title after insertion to reflect the conversation_id"""
    if not target._title_generated:
        target.generate_title()
        target._title_generated = True