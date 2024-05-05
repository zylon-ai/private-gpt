from typing import List
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from private_gpt.users.db.base_class import Base
from private_gpt.users.schemas.user import User

class Company(Base):
    """Models a Company table."""

    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)

    subscriptions = relationship("Subscription", back_populates="company")
    users = relationship("User", back_populates="company") 
    user_roles = relationship("UserRole", back_populates="company")
    departments = relationship("Department", back_populates="company")