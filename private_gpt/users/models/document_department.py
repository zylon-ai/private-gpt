from private_gpt.users.db.base_class import Base
from sqlalchemy import Column, Integer, Table, ForeignKey

document_department_association = Table(
    "document_department_association",
    Base.metadata,
    Column("department_id", Integer, ForeignKey("departments.id", ondelete="CASCADE", onupdate="CASCADE")),
    Column("document_id", Integer, ForeignKey("document.id", ondelete="CASCADE", onupdate="CASCADE")),
    extend_existing=True
)