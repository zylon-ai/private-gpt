from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String

from private_gpt.users.db.base_class import Base


class Department(Base):
    """Models a Department table."""

    __tablename__ = "departments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)

    company_id = Column(Integer, ForeignKey('companies.id'))
    company = relationship("Company", back_populates="departments")
