from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from private_gpt.users.db.base_class import Base

class Company(Base):
    """Models a Company table."""

    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    subscriptions = relationship("Subscription", back_populates="company")

    