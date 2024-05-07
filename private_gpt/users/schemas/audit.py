from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class AuditBase(BaseModel):
    id: int
    model: Optional[str] = None
    user_id: Optional[int] = None
    action: Optional[str] = None
    details: Optional[dict] = None
    ip_address: Optional[str] = None
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
    model: Optional[str] = None
    username: Optional[str] = None
    action: Optional[str] = None
    details: Optional[dict] = None
    timestamp: Optional[datetime]= None
    ip_address: Optional[str] = None


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