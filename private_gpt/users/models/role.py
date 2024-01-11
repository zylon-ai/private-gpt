from private_gpt.users.db.base_class import Base
from sqlalchemy import Column, String, Text, Integer

class Role(Base):
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), index=True)
    description = Column(Text)