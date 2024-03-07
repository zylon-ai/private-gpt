from datetime import datetime
from sqlalchemy import event, select, func, update
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Table
from private_gpt.users.models.department import Department
from private_gpt.users.db.base_class import Base

from private_gpt.users.models.document_department import document_department_association


class Document(Base):
    """Models a document table"""
    __tablename__ = "document"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(225), nullable=False, unique=True)
    uploaded_by = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
    )
    uploaded_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    uploaded_by_user = relationship(
        "User", back_populates="uploaded_documents")

    # Use document_department_association as the secondary for the relationship
    departments = relationship(
        "Department",
        secondary=document_department_association,
        back_populates="documents"
    )

# Event listeners for updating total_documents in Department
@event.listens_for(Document, 'after_insert')
@event.listens_for(Document, 'after_delete')
def update_total_documents(mapper, connection, target):
    total_documents = connection.execute(
        select([func.count()]).select_from(document_department_association).where(
            document_department_association.c.document_id == target.id)
    ).scalar()

    department_ids = [assoc.department_id for assoc in connection.execute(
        select([document_department_association.c.department_id]).where(
            document_department_association.c.document_id == target.id)
    )]

    # Update total_documents for each associated department
    for department_id in department_ids:
        connection.execute(
            update(Department).values(total_documents=total_documents).where(
                Department.id == department_id)
        )
