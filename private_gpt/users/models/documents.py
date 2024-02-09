from private_gpt.users.db.base_class import Base
from datetime import datetime, timedelta
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, Boolean, Float, ForeignKey, DateTime


class Documents(Base):
    """Models a user table"""
    _tablename_ = "document"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(225), nullable=False, unique=True)
    uploaded_by = Column(
        Integer,
        ForeignKey("users.id"),
        primary_key=True,
        nullable=False,
    )
    uploaded_at = Column(
        DateTime,
        default=datetime.utcnow,
    )
    uploaded_by_user = relationship("User", back_populates="uploaded_documents")