from sqlalchemy.orm import Session
from private_gpt.users.schemas.department import DepartmentCreate, DepartmentUpdate
from private_gpt.users.models.department import Department
from private_gpt.users.crud.base import CRUDBase
from typing import Optional


class CRUDDepartments(CRUDBase[Department, DepartmentCreate, DepartmentUpdate]):
    def get_by_id(self, db: Session, *, id: str) -> Optional[Department]:
        return db.query(self.model).filter(Department.id == id).first()

    def get_by_department_name(self, db: Session, *, name: str) -> Optional[Department]:
        return db.query(self.model).filter(Department.name == name).first()

department = CRUDDepartments(Department)
