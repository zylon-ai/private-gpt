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
from sqlalchemy import event, func, select, update
from sqlalchemy.orm import relationship
from private_gpt.users.db.base_class import Base
from private_gpt.users.models.department import Department
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

    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True)  
    company = relationship("Company", back_populates="users") 
    
    uploaded_documents = relationship("Document", back_populates="uploaded_by_user")

    user_role = relationship(
        "UserRole", back_populates="user", uselist=False, cascade="all, delete-orphan")
    
    department_id = Column(Integer, ForeignKey(
        "departments.id"), nullable=False)
    department = relationship("Department", back_populates="users")

    def __repr__(self):
        """Returns string representation of model instance"""
        return "<User {fullname!r}>".format(fullname=self.fullname)
    
    __table_args__ = (
        UniqueConstraint('fullname', name='unique_username_no_spacing'),
    )


@event.listens_for(User, 'after_insert')
@event.listens_for(User, 'after_delete')
def update_total_users(mapper, connection, target):
    department_id = target.department_id
    print(f"Department ID is: {department_id}")
    total_users = connection.execute(
        select([func.count()]).select_from(User).where(
            User.department_id == department_id)
    ).scalar()
    print(f"Total users is: {total_users}")
    connection.execute(
        update(Department).values(total_users=total_users).where(
            Department.id == department_id)
    )


