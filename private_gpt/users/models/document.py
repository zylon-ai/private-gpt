from private_gpt.users.db.base_class import Base

from datetime import datetime
from sqlalchemy import event, select, func, update
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from private_gpt.users.models.department import Department


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
    
    department_id = Column(Integer, ForeignKey(
        "departments.id"), nullable=False)

    department = relationship("Department", back_populates="documents")


@event.listens_for(Document, 'after_insert')
@event.listens_for(Document, 'after_delete')
def update_total_documents(mapper, connection, target):
    department_id = target.department_id
    print(f"Department ID is: {department_id}")

    # Use SQLAlchemy's ORM constructs for better readability and maintainability:
    total_documents = connection.execute(
        select([func.count()]).select_from(Document).where(
            Document.department_id == department_id)
    ).scalar()

    print(f"Total documents is: {total_documents}")
    print("Updating total documents")

    # Use the correct update construct for SQLAlchemy:
    connection.execute(
        update(Department).values(total_documents=total_documents).where(
            Department.id == department_id)
    )
