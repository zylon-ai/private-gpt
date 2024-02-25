from sqlalchemy.orm import Session
from private_gpt.users.schemas.department import DepartmentCreate, DepartmentUpdate
from private_gpt.users.models.department import Department
from private_gpt.users.crud.base import CRUDBase
from typing import Optional, List


class CRUDDepartments(CRUDBase[Department, DepartmentCreate, DepartmentUpdate]):
    def get_by_id(self, db: Session, *, id: int) -> Optional[Department]:
        return db.query(self.model).filter(Department.id == id).first()

    def get_by_department_name(self, db: Session, *, name: str) -> Optional[Department]:
        return db.query(self.model).filter(Department.name == name).first()

    def get_multi_department(
        self, db: Session, *, department_id: int, skip: int = 0, limit: int = 100
    ) -> List[Department]:
        return (
            db.query(self.model)
            .filter(Department.department_id == department_id)
            .offset(skip)
            .limit(limit)
            .all()
        )


department = CRUDDepartments(Department)
