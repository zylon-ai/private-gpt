from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    Boolean,
    UniqueConstraint,
    ForeignKey,
    DateTime
)
from sqlalchemy.orm import relationship
from sqlalchemy import event, func, select, update

from private_gpt.users.db.base_class import Base
from private_gpt.users.models.department import Department


class User(Base):
    """Models a user table"""
    __tablename__ = "users"
    id = Column(Integer, nullable=False, primary_key=True)

    email = Column(String(225), nullable=False, unique=True)
    hashed_password = Column(String, nullable=False)
    username = Column(String(225), nullable=False, unique=True)

    is_active = Column(Boolean, default=False)

    last_login = Column(DateTime, nullable=True, default=None)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
    )

    password_created = Column(DateTime, nullable=True)
    checker = Column(Boolean, default=False)

    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True)
    company = relationship("Company", back_populates="users")

    uploaded_documents = relationship(
        "Document", back_populates="uploaded_by_user",
        foreign_keys="[Document.uploaded_by]")

    user_role = relationship(
        "UserRole", back_populates="user", uselist=False, cascade="all, delete-orphan")

    department_id = Column(
        Integer, ForeignKey("departments.id"), nullable=False)
    
    department = relationship("Department", back_populates="users")

    __table_args__ = (
        UniqueConstraint('username', name='unique_username_no_spacing'),
    )

    def __repr__(self):
        """Returns string representation of model instance"""
        return "<User {username!r}>".format(username=self.username)


# Event listeners
# @event.listens_for(User, 'after_insert')
# @event.listens_for(User, 'after_delete')
# def update_total_users(mapper, connection, target):
#     department_id = target.department_id
#     total_users = connection.execute(
#         select([func.count()]).select_from(User).where(
#             User.department_id == department_id)
#     ).scalar()
#     connection.execute(
#         update(Department).values(total_users=total_users).where(
#             Department.id == department_id)
#     )


@event.listens_for(User, 'before_insert')
def set_password_created(mapper, connection, target):
    target.password_created = datetime.utcnow()


@event.listens_for(User, 'before_update', propagate=True)
def check_password_expiry(mapper, connection, target):
    if target.password_created and (
            datetime.now() - target.password_created).days > 90:
        target.is_active = False
    else:
        target.is_active = True
