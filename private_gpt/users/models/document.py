from datetime import datetime
from sqlalchemy import Boolean, event, select, func, update, insert
from sqlalchemy.orm import relationship, backref
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime

from private_gpt.users.models.department import Department
from private_gpt.users.models.makerchecker import MakerChecker
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
    is_enabled = Column(Boolean, default=True)
    # Use document_department_association as the secondary for the relationship
    verified = Column(Boolean, default=False)  # Added verified column
    departments = relationship(
        "Department",
        secondary=document_department_association,
        back_populates="documents"
    )
    # Relationship with MakerChecker
    maker_checker_entry = relationship(
        "MakerChecker",
        backref=backref("document", uselist=False),
        foreign_keys="[MakerChecker.record_id]",
        primaryjoin="and_(MakerChecker.table_name=='document', MakerChecker.record_id==Document.id)",
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
