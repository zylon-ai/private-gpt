from sqlalchemy.orm import Session
from private_gpt.users.schemas.documents import DocumentCreate, DocumentUpdate
from private_gpt.users.models.document import Document
from private_gpt.users.crud.base import CRUDBase
from typing import Optional, List


class CRUDDocuments(CRUDBase[Document, DocumentCreate, DocumentUpdate]):
    def get_by_id(self, db: Session, *, id: int) -> Optional[Document]:
        return db.query(self.model).filter(Document.id == id).first()

    def get_by_filename(self, db: Session, *, file_name: str) -> Optional[Document]:
        return db.query(self.model).filter(Document.filename == file_name).first()
    
    def get_multi_documents(
        self, db: Session, *,department_id: int, skip: int = 0, limit: int = 100
    ) -> List[Document]:
        return (
            db.query(self.model)
            .filter(Document.department_id == department_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

documents = CRUDDocuments(Document)
