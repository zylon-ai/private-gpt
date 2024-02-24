from typing import Optional

from private_gpt.users.crud.base import CRUDBase
from private_gpt.users.models.audit import Audit
from private_gpt.users.schemas.audit import AuditCreate, AuditUpdate
from sqlalchemy.orm import Session


class CRUDAudit(CRUDBase[Audit, AuditCreate, AuditUpdate]):
    def get_by_id(self, db: Session, *, id: str) -> Optional[Audit]:
        return db.query(self.model).filter(Audit.id == id).first()


audit = CRUDAudit(Audit)
