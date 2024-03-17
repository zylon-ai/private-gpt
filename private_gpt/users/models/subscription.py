from sqlalchemy import Column, Integer, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
from private_gpt.users.db.base_class import Base


class Subscription(Base):
    """Models a Subscription table."""
    __tablename__ = "subscriptions"

    sub_id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"))
    start_date = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime, default=lambda: datetime.utcnow() + timedelta(days=30)) 

    company = relationship("Company", back_populates="subscriptions")

    @property
    def is_active(self) -> bool:
        """Check if the subscription is active based on the end_date."""
        return self.end_date >= datetime.utcnow()
