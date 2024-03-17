# from sqlalchemy import Enum
# from datetime import datetime
# from enum import Enum as PythonEnum
# from private_gpt.users.db.base_class import Base
# from sqlalchemy import Column, Integer, ForeignKey, DateTime

# class MakerCheckerStatus(PythonEnum):
#     PENDING = 'pending'
#     APPROVED = 'approved'
#     REJECTED = 'rejected'


# class MakerCheckerActionType(PythonEnum):
#     INSERT = 'insert'
#     UPDATE = 'update'
#     DELETE = 'delete'


# class MakerChecker(Base):
#     """Models a maker-checker base"""

#     action_type = Column(Enum(MakerCheckerActionType), nullable=False, default=MakerCheckerActionType.INSERT)  # 'insert' or 'update' or 'delete'
#     status = Column(Enum(MakerCheckerStatus), nullable=False, default=MakerCheckerStatus.PENDING)  # 'pending', 'approved', or 'rejected'

#     verified_at = Column(DateTime, nullable=True)
#     verified_by = Column(Integer, ForeignKey("users.id"), nullable=True)

