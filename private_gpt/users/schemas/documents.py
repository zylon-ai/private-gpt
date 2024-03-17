from pydantic import BaseModel
from datetime import datetime
from typing import List
from fastapi import Form, UploadFile, File

class DocumentsBase(BaseModel):
    filename: str

class DepartmentList(BaseModel):
    id: int
    name: str

class DocumentCreate(DocumentsBase):
    uploaded_by: int

class DocumentUpdate(BaseModel):
    id: int
    status: str
    is_enabled: bool

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
    uploaded_by: int
    uploaded_at: datetime
    departments: List[DepartmentList] = []

    class Config:
        orm_mode = True

class DocumentMakerChecker(DocumentCreate):
    action_type: str
    status: str
    doc_type_id: int

class DocumentMakerCreate(DocumentMakerChecker):
    pass


class DocumentCheckerUpdate(BaseModel):
    status: str
    is_enabled: bool
    verified_at: datetime
    verified_by: int


class DocumentDepartmentList(BaseModel):
    departments_ids: str = Form(...)
    doc_type_id: int
