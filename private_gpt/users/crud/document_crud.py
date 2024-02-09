from sqlalchemy.orm import Session
from private_gpt.users.schemas.documents import DocumentCreate, DocumentUpdate
from private_gpt.users.models.documents import Documents
from private_gpt.users.crud.base import CRUDBase
from typing import Optional


class CRUDDocuments(CRUDBase[Documents, DocumentCreate, DocumentUpdate]):
    def get_by_id(self, db: Session, *, id: str) -> Optional[Documents]:
        return db.query(self.model).filter(Documents.id == id).first()

    def get_by_filename(self, db: Session, *, file_name: str) -> Optional[Documents]:
        return db.query(self.model).filter(Documents.filename == file_name).first()


documents = CRUDDocuments(Documents)
