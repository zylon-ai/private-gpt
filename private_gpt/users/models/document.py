from typing import Optional
from fastapi_filter.contrib.sqlalchemy import Filter
from datetime import datetime
from sqlalchemy.orm import relationship
from sqlalchemy import Boolean, event, select, func, update
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime

from private_gpt.users.db.base_class import Base
from private_gpt.users.models.document_department import document_department_association
from sqlalchemy import Enum
from enum import Enum as PythonEnum

class MakerCheckerStatus(PythonEnum):
    PENDING = 'PENDING'
    APPROVED = 'APPROVED'
    REJECTED = 'REJECTED'


class MakerCheckerActionType(PythonEnum):
    INSERT = 'INSERT'
    UPDATE = 'UPDATE'
    DELETE = 'DELETE'

class DocumentType(Base):
    """Models a document table"""
    __tablename__ = "document_type"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(String(225), nullable=False, unique=True)
    documents = relationship("Document", back_populates='doc_type')


class Document(Base):
    """Models a document table"""
    __tablename__ = "document"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(225), nullable=False, unique=True)
    uploaded_by = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False
    )
    uploaded_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    uploaded_by_user = relationship(
        "User", back_populates="uploaded_documents",
        foreign_keys="[Document.uploaded_by]")
    
    is_enabled = Column(Boolean, default=True)
    verified = Column(Boolean, default=False) 
    
    doc_type_id = Column(Integer, ForeignKey("document_type.id"))
    doc_type = relationship("DocumentType", back_populates='documents')

    action_type = Column(Enum(MakerCheckerActionType), nullable=False,
                         default=MakerCheckerActionType.INSERT)  # 'insert' or 'update' or 'delete'
    # 'pending', 'approved', or 'rejected'
    status = Column(Enum(MakerCheckerStatus), nullable=False,
                    default=MakerCheckerStatus.PENDING)

    verified_at = Column(DateTime, nullable=True)
    verified_by = Column(Integer, ForeignKey("users.id"), nullable=True)


    departments = relationship(
        "Department",
        secondary=document_department_association,
        back_populates="documents"
    )
    
# @event.listens_for(Document, 'after_insert')
# @event.listens_for(Document, 'after_delete')
# def update_total_documents(mapper, connection, target):
#     total_documents = connection.execute(
#         func.count().select().select_from(document_department_association).where(
#             document_department_association.c.document_id == target.id)
#     ).scalar()

#     department_ids = [assoc.department_id for assoc in connection.execute(
#         select([document_department_association.c.department_id]).where(
#             document_department_association.c.document_id == target.id)
#     )]

#     # Update total_documents for each associated department
#     for department_id in department_ids:
#         connection.execute(
#             update(Department).values(total_documents=total_documents).where(
#                 Department.id == department_id)
#         )

