from typing import Optional

from private_gpt.users.crud.base import CRUDBase
from private_gpt.users.models.company import Company
from private_gpt.users.schemas.company import CompanyCreate, CompanyUpdate
from sqlalchemy.orm import Session


class CRUDCompany(CRUDBase[Company, CompanyCreate, CompanyUpdate]):
    def get_by_id(self, db: Session, *, id: str) -> Optional[Company]:
        return db.query(self.model).filter(Company.id == id).first()

company = CRUDCompany(Company)