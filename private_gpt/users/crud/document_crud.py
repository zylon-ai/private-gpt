from sqlalchemy.sql.expression import desc, asc
from sqlalchemy import or_, and_
from sqlalchemy.orm import Session
from private_gpt.users.schemas.documents import DocumentCreate, DocumentUpdate
from private_gpt.users.models.document import Document
from private_gpt.users.models.department import Department
from private_gpt.users.models.document_department import document_department_association
from private_gpt.users.crud.base import CRUDBase
from typing import Optional, List


class CRUDDocuments(CRUDBase[Document, DocumentCreate, DocumentUpdate]):
    def get_by_id(self, db: Session, *, id: int) -> Optional[Document]:
        return db.query(self.model).filter(Document.id == id).first()

    def get_by_filename(self, db: Session, *, file_name: str) -> Optional[Document]:
        return db.query(self.model).filter(Document.filename == file_name).first()
    
    def get_multi_documents(
        self, db: Session, *, skip: int = 0, limit: int = 100
    ) -> List[Document]:
        return (
            db.query(self.model)
            .order_by(desc(getattr(Document, 'uploaded_at')))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_documents_by_departments(
            self, db: Session, *, department_id: int, skip: int = 0, limit: int = 100
        ) -> List[Document]:
            return (
                db.query(self.model)
                .join(document_department_association)
                .join(Department)
                .filter(document_department_association.c.department_id == department_id)
                .offset(skip)
                .limit(limit)
                .all().order_by(desc(getattr(Document, 'uploaded_at')))
            )
    
    def get_files_to_verify(
            self, db: Session, *, skip: int = 0, limit: int = 100
        ) -> List[Document]:
            return (
                db.query(self.model)
                .filter(Document.status == 'PENDING')
                .offset(skip)
                .limit(limit)
                .all()
            )
    
    def get_enabled_documents_by_departments(
            self, db: Session, *, department_id: int, skip: int = 0, limit: int = 100
        ) -> List[Document]:
            all_department_id = 1 # department ID for "ALL" is 1

            return (
                db.query(self.model)
                .join(document_department_association)
                .join(Department)
                .filter(
                    or_(
                        and_(
                            document_department_association.c.department_id == department_id,
                            Document.is_enabled == True,
                        ),
                        and_(
                            document_department_association.c.department_id == all_department_id,
                            Document.is_enabled == True,
                        ),
                    )
                )
                .offset(skip)
                .limit(limit)
                .all()
            )

    def filter_query(
        self, db: Session, *,
        filename: Optional[str] = None,
        uploaded_by: Optional[str] = None,
        action_type: Optional[str] = None,
        status: Optional[str] = None,
        order_by: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Document]:
        query = db.query(Document)
        if filename:
            query = query.filter(
                Document.filename.ilike(f"%{filename}%"))
        if uploaded_by:
            query = query.filter(
                Document.uploaded_by == uploaded_by)
        if action_type:
            query = query.filter(
                Document.action_type == action_type)
        if status:
            query = query.filter(Document.status == status)
        if order_by == "desc":
            query = query.order_by(desc(getattr(Document, 'uploaded_at')))
        else:
            query = query.order_by(asc(getattr(Document, 'uploaded_at')))

        return query.offset(skip).limit(limit).all()

documents = CRUDDocuments(Document)
