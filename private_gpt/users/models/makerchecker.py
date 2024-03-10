from datetime import datetime
from sqlalchemy import Boolean, event, select, func, update
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from private_gpt.users.db.base_class import Base
from sqlalchemy import Enum
from enum import Enum as PythonEnum

class MakerCheckerStatus(PythonEnum):
    PENDING = 'pending'
    APPROVED = 'approved'
    REJECTED = 'rejected'


class MakerCheckerActionType(PythonEnum):
    INSERT = 'insert'
    UPDATE = 'update'


class MakerChecker(Base):
    """Models a maker-checker table"""
    __tablename__ = "maker_checker"

    id = Column(Integer, primary_key=True, index=True)
    table_name = Column(String(50), nullable=False)
    record_id = Column(Integer, nullable=False)
    action_type = Column(Enum(MakerCheckerActionType), nullable=False, default=MakerCheckerActionType.INSERT)  # 'insert' or 'update'
    status = Column(Enum(MakerCheckerStatus), nullable=False, default=MakerCheckerStatus.PENDING)  # 'pending', 'approved', or 'rejected'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    verified_at = Column(DateTime, nullable=True)
    verified_by = Column(Integer, ForeignKey("users.id"), nullable=True)

    def __repr__(self):
        return f"<MakerChecker {self.table_name} {self.record_id}>"
