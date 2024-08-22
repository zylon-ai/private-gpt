from sqlalchemy.orm import Session
from private_gpt.users.schemas.category import CategoryCreate, CategoryUpdate
from private_gpt.users.models.category import Category
from private_gpt.users.crud.base import CRUDBase
from typing import Optional, List

class CRUDCategory(CRUDBase[Category, CategoryCreate, CategoryUpdate]):
    def get_by_id(self, db: Session, *, id: int) -> Optional[Category]:
        return db.query(self.model).filter(Category.id == id).first()

    def get_by_category_name(self, db: Session, *, name: str) -> Optional[Category]:
        return db.query(self.model).filter(Category.name == name).first()
    
category = CRUDCategory(Category)
