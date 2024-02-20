from sqlalchemy.orm import Session
from private_gpt.users.schemas.documents import DocumentCreate, DocumentUpdate
from private_gpt.users.models.document import Document
from private_gpt.users.crud.base import CRUDBase
from typing import Optional


class CRUDDocuments(CRUDBase[Document, DocumentCreate, DocumentUpdate]):
    def get_by_id(self, db: Session, *, id: str) -> Optional[Document]:
        return db.query(self.model).filter(Document.id == id).first()

    def get_by_filename(self, db: Session, *, file_name: str) -> Optional[Document]:
        return db.query(self.model).filter(Document.filename == file_name).first()


documents = CRUDDocuments(Document)
