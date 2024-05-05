from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class AuditBase(BaseModel):
    id: int
    model: str
    user_id: int
    action: str
    details: dict
    ip_address: str
    timestamp: Optional[datetime]


class AuditCreate(AuditBase):
    pass


class AuditUpdate(AuditBase):
    id: int


class AuditInDB(AuditBase):
    id: int

    class Config:
        orm_mode = True

class Audit(BaseModel):
    id: int
    model: str
    username: str
    action: str
    details: dict
    timestamp: Optional[datetime]
    ip_address: str


class GetAudit(BaseModel):
    id: int


class AuditFilter(BaseModel):
    skip: int = 0,
    limit: int = 100,
    model: Optional[str] = None
    username: Optional[str] = None
    action: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class ExcelFilter(BaseModel):
    model: Optional[str] = None
    username: Optional[str] = None
    action: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None