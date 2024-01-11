from private_gpt.users.core.config import SQLALCHEMY_DATABASE_URI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(SQLALCHEMY_DATABASE_URI, echo=True, future=True, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# test_engine = create_engine(
#     f"{settings.SQLALCHEMY_DATABASE_URI}_test", pool_pre_ping=True
# )
# TestingSessionLocal = sessionmaker(
#     autocommit=False, autoflush=False, bind=test_engine
# )