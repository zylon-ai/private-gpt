import datetime
from sqlalchemy import (
    Column, 
    String, 
    Integer,
    Boolean, 
    UniqueConstraint, 
    PrimaryKeyConstraint,
    DateTime,
    ForeignKey
)
from sqlalchemy.orm import relationship, backref
from sqlalchemy import event, func, select, update, insert

from private_gpt.users.db.base_class import Base
from private_gpt.users.models.department import Department
from private_gpt.users.models.makerchecker import MakerChecker


class User(Base):
    """Models a user table"""
    __tablename__ = "users"
    id = Column(Integer, nullable=False, primary_key=True)
    
    email = Column(String(225), nullable=False, unique=True)
    hashed_password = Column(String, nullable=False)
    fullname = Column(String(225), nullable=False, unique=True)

    UniqueConstraint("email", name="uq_user_email")
    PrimaryKeyConstraint("id", name="pk_user_id")

    is_active = Column(Boolean, default=False)

    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )
    
    password_created = Column(DateTime, nullable=True)
    checker = Column(Boolean, default=False)

    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True)  
    company = relationship("Company", back_populates="users") 
    
    uploaded_documents = relationship("Document", back_populates="uploaded_by_user")

    user_role = relationship(
        "UserRole", back_populates="user", uselist=False, cascade="all, delete-orphan")
    
    department_id = Column(Integer, ForeignKey(
        "departments.id"), nullable=False)
    department = relationship("Department", back_populates="users")

    maker_checker_entry = relationship(
        "MakerChecker",
        backref=backref("user", uselist=False),
        foreign_keys="[MakerChecker.record_id]",
        primaryjoin="and_(MakerChecker.table_name=='users', MakerChecker.record_id==User.id)",
    )

    def __repr__(self):
        """Returns string representation of model instance"""
        return "<User {fullname!r}>".format(fullname=self.fullname)
    
    __table_args__ = (
        UniqueConstraint('fullname', name='unique_username_no_spacing'),
    )


@event.listens_for(User, 'after_insert')
def create_maker_checker_entry(mapper, connection, target):
    # Create a MakerChecker entry for the new User record
    connection.execute(
        insert(MakerChecker).values(
            table_name='users',
            record_id=target.id,
            action_type='insert',
            status='pending',
            verified_at=None,
            verified_by=None,
        )
    )


@event.listens_for(User, 'after_insert')
@event.listens_for(User, 'after_delete')
def update_total_users(mapper, connection, target):
    department_id = target.department_id
    total_users = connection.execute(
        select([func.count()]).select_from(User).where(
            User.department_id == department_id)
    ).scalar()
    connection.execute(
        update(Department).values(total_users=total_users).where(
            Department.id == department_id)
    )


@event.listens_for(User, 'before_insert')
def set_password_created(mapper, connection, target):
    target.password_created = datetime.datetime.utcnow()
    connection.execute(
        update(User)
        .values(password_created=datetime.datetime.utcnow())
        .where(User.id == target.id)
    )

@event.listens_for(User, 'before_update', propagate=True)
def check_password_expiry(mapper, connection, target):
    if target.password_created and (
            datetime.datetime.utcnow() - target.password_created).days > 90:
        target.is_active = False
        connection.execute(
            update(User)
            .values(is_active=False)
            .where(User.id == target.id)
        )
    else:
        connection.execute(
            update(User)
            .values(is_active=True)
            .where(User.id == target.id)
        )

