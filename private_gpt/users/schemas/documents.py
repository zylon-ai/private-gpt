from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class DocumentsBase(BaseModel):
    filename: str


class DocumentCreate(DocumentsBase):
    uploaded_by: int


class DocumentUpdate(DocumentsBase):
    pass


class Document(DocumentsBase):
    id: int
    uploaded_by: int
    uploaded_at: datetime

    class Config:
        orm_mode = True
