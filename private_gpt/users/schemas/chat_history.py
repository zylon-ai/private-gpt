from typing import Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime


class ChatBase(BaseModel):
    title: Optional[str]

class ChatCreate(ChatBase):
    user_id: int
    messages: Optional[Dict[str, Any]]

class ChatUpdate(ChatBase):
    conversation_id: int
    messages: Dict


class ChatDelete(BaseModel):
    conversation_id: int

class ChatMessages(BaseModel):
    messages: Dict[str, Any]

class Chat(ChatBase):
    conversation_id: int
    created_at: datetime
    user_id: int

    class Config:
        orm_mode = True
