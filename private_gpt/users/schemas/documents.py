from pydantic import BaseModel
from datetime import datetime
from typing import List

class DocumentsBase(BaseModel):
    filename: str

class DepartmentList(BaseModel):
    id: int
    name: str

class DocumentCreate(DocumentsBase):
    uploaded_by: int

class DocumentUpdate(DocumentsBase):
    pass


class DocumentList(DocumentsBase):
    id: int
    uploaded_by: int
    uploaded_at: datetime
    departments: List[DepartmentList] = []

class Document(DocumentsBase):
    id: int
    filename: str
    uploaded_by: int
    uploaded_at: datetime
    departments: List[DepartmentList] = []

    class Config:
        orm_mode = True
