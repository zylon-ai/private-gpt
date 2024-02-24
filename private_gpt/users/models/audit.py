from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from private_gpt.users.db.base_class import Base
from sqlalchemy.dialects.postgresql import JSONB  


class Audit(Base):
    __tablename__ = "audit"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    model = Column(String, nullable=False)
    action = Column(String, nullable=False)
    details = Column(JSONB, nullable=True)

    def __repr__(self):
        return f"<Audit(id={self.id}, timestamp={self.timestamp}, user_id={self.user_id}, model={self.model}, action={self.action}, details={self.details})>"
