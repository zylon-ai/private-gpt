from private_gpt.users.db.session import SessionLocal

from sqlalchemy.orm import Session

def init_db(db: Session) -> None:
    """Database session generator"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
