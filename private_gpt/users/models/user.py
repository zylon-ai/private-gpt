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
from sqlalchemy.orm import relationship
from private_gpt.users.db.base_class import Base

class User(Base):
    """Models a user table"""
    __tablename__ = "users"
    id = Column(Integer, nullable=False, primary_key=True)
    
    email = Column(String(225), nullable=False, unique=True)
    hashed_password = Column(String, nullable=False)
    fullname = Column(String(225), nullable=False)

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

    user_role = relationship("UserRole", back_populates="user", uselist=False)

    def __repr__(self):
        """Returns string representation of model instance"""
        return "<User {fullname!r}>".format(fullname=self.fullname)
    
    