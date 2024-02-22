from typing import Any, Dict, List, Optional, Union

from private_gpt.users.core.security import get_password_hash, verify_password
from private_gpt.users.crud.base import CRUDBase
from private_gpt.users.models.user import User
from private_gpt.users.schemas.user import UserCreate, UserUpdate
from private_gpt.users.models.user_role import UserRole
from private_gpt.users.models.role import Role
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload


class CRUDUser(CRUDBase[User, UserCreate, UserUpdate]):
    def get_by_email(self, db: Session, *, email: str) -> Optional[User]:
        return db.query(self.model).filter(User.email == email).first()
    
    def create(self, db: Session, *, obj_in: UserCreate) -> User:
        db_obj = User(
            email=obj_in.email,
            hashed_password=get_password_hash(obj_in.password),
            fullname=obj_in.fullname,
            company_id=obj_in.company_id,
            department_id=obj_in.department_id,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self,
        db: Session,
        *,
        db_obj: User,
        obj_in: Union[UserUpdate, Dict[str, Any]],
    ) -> User:
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        if "password" in update_data:
            hashed_password = get_password_hash(update_data["password"])
            del update_data["password"]
            update_data["hashed_password"] = hashed_password
        return super().update(db, db_obj=db_obj, obj_in=update_data)

    def get_multi(
        self, db: Session, *, skip: int = 0, limit: int = 100,
    ) -> List[User]:
        return db.query(self.model).offset(skip).limit(limit).all()

    def authenticate(
        self, db: Session, *, email: str, password: str
    ) -> Optional[User]:
        user = self.get_by_email(db, email=email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    def is_active(self, user: User) -> bool:
        return user.is_active

    def get_by_account_id(
        self,
        db: Session,
        *,
        account_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> List[User]:
        return (
            db.query(self.model)
            .filter(User.account_id == account_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_multi_by_company_id(
        self, db: Session, *, company_id: str, skip: int = 0, limit: int = 100
    ) -> List[User]:
        return (
            db.query(self.model)
            .join(User.user_role)
            .filter(UserRole.company_id == company_id)
            .options(joinedload(User.user_role).joinedload(UserRole.role))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_by_name(self, db: Session, *, name: str) -> Optional[User]:
        return db.query(self.model).filter(User.fullname == name).first()

    def get_by_department_id(
        self, db: Session, *, department_id: int, skip: int = 0, limit: int = 100
    ) -> List[User]:
        return (
            db.query(self.model)
            .filter(User.department_id == department_id)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_by_id(self, db: Session, *, id: int) -> Optional[User]:
        return db.query(self.model).filter(User.id == id).first()
    
user = CRUDUser(User)
