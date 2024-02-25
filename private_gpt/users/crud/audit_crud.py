from typing import Optional, List

from private_gpt.users.crud.base import CRUDBase
from private_gpt.users.models.audit import Audit
from private_gpt.users.schemas.audit import AuditCreate, AuditUpdate
from sqlalchemy import desc
from sqlalchemy.orm import Session


class CRUDAudit(CRUDBase[Audit, AuditCreate, AuditUpdate]):
    def get_multi_desc(
        self, db: Session, *, skip: int = 0, limit: int = 100
    ) -> List[Audit]:
        return db.query(self.model).order_by(desc(self.model.timestamp)).offset(skip).limit(limit).all()
    
    def get_by_id(self, db: Session, *, id: str) -> Optional[Audit]:
        return db.query(self.model).filter(Audit.id == id).first()


audit = CRUDAudit(Audit)
