from sqlalchemy import ForeignKey, event
from sqlalchemy.orm import relationship, Session
from sqlalchemy import Column, Integer, String

from private_gpt.users.db.base_class import Base
# from private_gpt.users.models.document import Document
# from private_gpt.users.models.user import User


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


# @event.listens_for(Department, 'after_insert')
# @event.listens_for(Department, 'after_update')
# def update_total_users(mapper, connection, target):
#     print("--------------------------------------------------------------Calling Event User------------------------------------------------------------------------")
#     connection.execute(
#         Department.__table__.update().
#         where(Department.id == target.id).
#         values(total_users=Session.object_session(target).query(
#             User).filter_by(department_id=target.id).count())
#     )


# @event.listens_for(Department, 'after_insert')
# @event.listens_for(Department, 'after_update')
# def update_total_documents(mapper, connection, target):
#     print("--------------------------------------------------------------Calling Event Department------------------------------------------------------------------------")
#     connection.execute(
#         Department.__table__.update().
#         where(Department.id == target.id).
#         values(total_documents=Session.object_session(target).query(
#             Document).filter_by(department_id=target.id).count())
#     )
