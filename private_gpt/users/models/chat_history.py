from datetime import datetime

from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON

from private_gpt.users.db.base_class import Base


class ChatHistory(Base):
    """Models a chat history table"""

    __tablename__ = "chat_history"

    id = Column(Integer, nullable=False, primary_key=True)
    title = Column(String(255), nullable=False)
    messages = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="chat_histories")

    def __init__(self, messages, user_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages = messages
        self.user_id = user_id
        self.title = self.generate_title()

    def generate_title(self):
        if self.messages:
            first_user_message = next(+
                (msg["message"]
                 for msg in self.messages if msg["sender"] == "user"), None
            )
            if first_user_message:
                return first_user_message[:30]
        return "Untitled Chat"

    def __repr__(self):
        """Returns string representation of model instance"""
        return f"<ChatHistory {self.id!r}>"
