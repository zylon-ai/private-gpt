from sqlalchemy import Column, Integer, String, Boolean, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
from fastapi import Depends
from private_gpt.users.db.base_class import Base


class Subscription(Base):
    """Models a Subscription table."""
    __tablename__ = "subscriptions"

    sub_id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"))
    start_date = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime, default=datetime.utcnow() + timedelta(days=30))  # Example: 30 days subscription period
    is_active = Column(Boolean, default=True)

    company = relationship("Company", back_populates="subscriptions")

