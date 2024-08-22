from sqlalchemy.sql.expression import desc, asc
from sqlalchemy import or_, and_
from sqlalchemy.orm import Session, joinedload
from private_gpt.users.schemas.documents import DocumentCreate, DocumentUpdate
from private_gpt.users.models.document import Document
from private_gpt.users.models.department import Department
from private_gpt.users.models.document_department import document_department_association
from private_gpt.users.models.category import document_category_association, Category
from private_gpt.users.crud.base import CRUDBase
from private_gpt.constants import ALL_DEPARTMENT

from typing import Optional, List


class CRUDDocuments(CRUDBase[Document, DocumentCreate, DocumentUpdate]):
    def get_by_id(self, db: Session, *, id: int) -> Optional[Document]:
        return db.query(self.model).filter(Document.id == id).first()

    def get_by_filename(self, db: Session, *, file_name: str) -> Optional[Document]:
        return db.query(self.model).filter(Document.filename == file_name).first()
    
    def get_multi_documents(
        self, db: Session,
    ) -> List[Document]:
        return (
            db.query(self.model)
            .options(joinedload(Document.categories))  
            .order_by(desc(getattr(Document, 'uploaded_at')))
            .all()
        )

    def get_documents_by_departments(
            self, db: Session, *, department_id: int
        ) -> List[Document]:
        all_department_id = ALL_DEPARTMENT
        return (
            db.query(self.model)
            .join(document_department_association)
            .join(Department)
            .options(joinedload(Document.categories))
            .filter(
                or_(
                    and_(
                        document_department_association.c.department_id == department_id,
                    ),
                    and_(
                        document_department_association.c.department_id == all_department_id,
                    ),
                )
            )
            .order_by(desc(getattr(Document, 'uploaded_at')))
            .all()
        )
    
    def get_files_to_verify(
            self, db: Session,
        ) -> List[Document]:
        return (
            db.query(self.model)
            .filter(Document.status == 'PENDING')
            .options(joinedload(Document.categories))
            .all()
        )
    
    def get_enabled_documents_by_departments(
            self, db: Session, *, department_id: int
        ) -> List[Document]:
        all_department_id = ALL_DEPARTMENT 

        return (
            db.query(self.model)
            .join(document_department_association)
            .join(Department)
            .options(joinedload(Document.categories))
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
            .all()
        )

    def get_documents_by_categories(
            self, db: Session, *, category_id: int
        ) -> List[Document]:
        return (
            db.query(self.model)
            .join(document_category_association)
            .join(Category)
            .filter(
                document_category_association.c.category_id == category_id
            )
            .order_by(desc(Document.uploaded_at))
            .all()
        )

    def filter_query(
        self, db: Session, *,
        filename: Optional[str] = None,
        uploaded_by: Optional[str] = None,
        action_type: Optional[str] = None,
        status: Optional[str] = None,
        category_id: Optional[int] = None,
        order_by: Optional[str] = None,
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
        if category_id:
            query = query.join(document_category_association).filter(
                document_category_association.c.category_id == category_id
            )
        if order_by == "desc":
            query = query.order_by(desc(Document.uploaded_at))
        else:
            query = query.order_by(asc(Document.uploaded_at))

        return query.all()

documents = CRUDDocuments(Document)
