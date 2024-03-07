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

class DocumentEnable(DocumentsBase):
    is_enabled: bool

class DocumentDepartmentUpdate(DocumentsBase):
    departments: List[int] = []

class DocumentList(DocumentsBase):
    id: int
    is_enabled: bool
    uploaded_by: int
    uploaded_at: datetime
    departments: List[DepartmentList] = []

class Document(BaseModel):
    id: int
    is_enabled: bool
    filename: str
    uploaded_by: str
    uploaded_at: datetime
    departments: List[DepartmentList] = []

    class Config:
        orm_mode = True
