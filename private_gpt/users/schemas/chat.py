from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class ChatItemBase(BaseModel):
    conversation_id: int
    sender: str
    content: Optional[str]


class ChatItemCreate(ChatItemBase):
    pass

class ChatItemUpdate(ChatItemBase):
    like: Optional[bool]


class ChatItem(ChatItemBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class ChatHistoryBase(BaseModel):
    user_id: int
    title: Optional[str]


class ChatHistoryCreate(ChatHistoryBase):
    chat_items: Optional[List[ChatItemCreate]]

class ChatHistoryUpdate(ChatHistoryBase):
    updated_at: datetime
    chat_items: Optional[List[ChatItemCreate]]

class Chat(BaseModel):
    conversation_id: int

class ChatHistory(ChatHistoryBase):
    conversation_id: int
    created_at: datetime
    updated_at: datetime
    chat_items: List[ChatItem]

    class Config:
        orm_mode = True

class ChatDelete(BaseModel):
    conversation_id: int

class CreateChatHistory(BaseModel):
    user_id: int
