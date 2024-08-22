
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String
from private_gpt.users.db.base_class import Base
from sqlalchemy import Column, Integer, Table, ForeignKey


class Category(Base):
    """Models a Category table."""
    
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)

    documents = relationship("Document", secondary="document_category_association", back_populates="categories")
    
document_category_association = Table(
    "document_category_association",
    Base.metadata,
    Column("document_id", Integer, ForeignKey("document.id", ondelete="CASCADE", onupdate="CASCADE")),
    Column("category_id", Integer, ForeignKey("categories.id", ondelete="CASCADE", onupdate="CASCADE")),
    extend_existing=True
)