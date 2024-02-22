from sqlalchemy import ForeignKey, event
from sqlalchemy.orm import relationship, Session
from sqlalchemy import Column, Integer, String

from private_gpt.users.db.base_class import Base
from private_gpt.users.models.document import Document
from private_gpt.users.models.user import User


class Department(Base):
    """Models a Department table."""

    __tablename__ = "departments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)

    company_id = Column(Integer, ForeignKey('companies.id'))
    company = relationship("Company", back_populates="departments")

    users = relationship("User", back_populates="department")
    documents = relationship("Document", back_populates="department")

    total_users = Column(Integer, default=0)
    total_documents = Column(Integer, default=0)



def update_total_users(mapper, connection, target):
    session = Session(bind=connection)
    target.total_users = session.query(User).filter_by(
        department_id=target.id).count()


def update_total_documents(mapper, connection, target):
    session = Session(bind=connection)
    target.total_documents = session.query(
        Document).filter_by(department_id=target.id).count()


# Attach event listeners to Department model
event.listen(Department, 'after_insert', update_total_users)
event.listen(Department, 'after_update', update_total_users)
event.listen(Department, 'after_insert', update_total_documents)
event.listen(Department, 'after_update', update_total_documents)
