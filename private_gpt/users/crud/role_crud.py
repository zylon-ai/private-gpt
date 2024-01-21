from typing import Optional

from private_gpt.users.crud.base import CRUDBase
from private_gpt.users.models.role import Role
from private_gpt.users.schemas.role import RoleCreate, RoleUpdate
from sqlalchemy.orm import Session


class CRUDRole(CRUDBase[Role, RoleCreate, RoleUpdate]):
    def get_by_name(self, db: Session, *, name: str) -> Optional[Role]:
        return db.query(self.model).filter(Role.name == name).first()
    
role = CRUDRole(Role)