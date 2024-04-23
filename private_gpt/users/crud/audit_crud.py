from typing import Optional, List

from private_gpt.users import crud
from private_gpt.users.crud.base import CRUDBase
from private_gpt.users.models.audit import Audit
from private_gpt.users.schemas.audit import AuditCreate, AuditUpdate, AuditFilter
from sqlalchemy import desc
from sqlalchemy.orm import Session


class CRUDAudit(CRUDBase[Audit, AuditCreate, AuditUpdate]):
    def get_multi_desc(
        self, db: Session, *, skip: int = 0, limit: int = 100
    ) -> List[Audit]:
        return db.query(self.model).order_by(desc(self.model.timestamp)).offset(skip).limit(limit).all()
    
    def filter(
        self, db: Session, *, obj_in : AuditFilter
    ) -> List[Audit]:
        
        def get_id(username):
            user = crud.user.get_by_name(db, name=username)
            if user:
                return user.id
            return None
        query = db.query(Audit)
        if obj_in.model:
            query = query.filter(Audit.model == obj_in.model)
        if obj_in.username:
            query = query.filter(Audit.user_id == get_id(obj_in.username))
        if obj_in.action:
            query = query.filter(Audit.action == obj_in.action)
        if obj_in.start_date:
            query = query.filter(Audit.timestamp >= obj_in.start_date)
        if obj_in.end_date:
            query = query.filter(Audit.timestamp <= obj_in.end_date)

        return query.order_by(desc(self.model.timestamp)).offset(obj_in.skip).limit(obj_in.limit).all()
        
        
    def excel_filter(
        self, db: Session, *, obj_in : AuditFilter
    ) -> List[Audit]:
        
        def get_id(username):
            user = crud.user.get_by_name(db, name=username)
            if user:
                return user.id
            return None
        query = db.query(Audit)
        if obj_in.model:
            query = query.filter(Audit.model == obj_in.model)
        if obj_in.username:
            query = query.filter(Audit.user_id == get_id(obj_in.username))
        if obj_in.action:
            query = query.filter(Audit.action == obj_in.action)
        if obj_in.start_date:
            query = query.filter(Audit.timestamp >= obj_in.start_date)
        if obj_in.end_date:
            query = query.filter(Audit.timestamp <= obj_in.end_date)

        return query.order_by(desc(self.model.timestamp)).all()
        

    def get_by_id(self, db: Session, *, id: str) -> Optional[Audit]:
        return db.query(self.model).filter(Audit.id == id).first()


audit = CRUDAudit(Audit)
