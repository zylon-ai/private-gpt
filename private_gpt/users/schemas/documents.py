from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class DocumentsBase(BaseModel):
    filename: str


class DocumentCreate(DocumentsBase):
    uploaded_by: int
    department_id: int


class DocumentUpdate(DocumentsBase):
    pass

class DocumentList(DocumentsBase):
    id: int
    uploaded_by: int
    uploaded_at: datetime


class Document(DocumentsBase):
    id: int
    uploaded_by: int
    uploaded_at: datetime
    department_id: int

    class Config:
        orm_mode = True
