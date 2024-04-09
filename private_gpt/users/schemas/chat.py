from datetime import datetime
from typing import List, Optional, Union, Dict
from pydantic import BaseModel, Json
import uuid

class ChatItemBase(BaseModel):
    conversation_id: uuid.UUID
    sender: str
    content: Union[str, Dict]


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
    conversation_id: uuid.UUID

class ChatHistory(ChatHistoryBase):
    conversation_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    chat_items: List[ChatItem]

    class Config:
        orm_mode = True

class ChatDelete(BaseModel):
    conversation_id: uuid.UUID

class CreateChatHistory(BaseModel):
    user_id: int
