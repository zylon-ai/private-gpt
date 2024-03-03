from private_gpt.users.core.db_config import SQLALCHEMY_DATABASE_URI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging

logging.basicConfig()
logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
logging.getLogger("sqlalchemy.pool").setLevel(logging.DEBUG)
engine = create_engine(SQLALCHEMY_DATABASE_URI, echo=True,
                       future=True, pool_pre_ping=True, logging_name="myengine")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
