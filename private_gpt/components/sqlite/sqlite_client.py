import logging
import threading
from importlib.util import find_spec
from pathlib import Path

from private_gpt.settings.settings import Settings

if find_spec("sqlalchemy") is None:
    raise ImportError("SQLAlchemy dependency not found")

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SQLiteClient:
    def __init__(self, local_path: str, database: str) -> None:
        self._local_path = Path(local_path).joinpath(f"{database}.db")
        Path(self._local_path).parent.mkdir(parents=True, exist_ok=True)
        self._sync_engine = create_engine(f"sqlite:///{self._local_path}")
        self._sync_engine = self._sync_engine.execution_options(
            schema_translate_map={"app": None}
        )
        self._async_engine = create_async_engine(
            f"sqlite+aiosqlite:///{self._local_path}"
        )
        self._async_engine = self._async_engine.execution_options(
            schema_translate_map={"app": None}
        )
        self.sync_session = self.new_sync_session()
        self.async_session = self.new_async_session()

    def new_sync_session(self) -> sessionmaker[Session]:
        return sessionmaker(self._sync_engine)

    def new_async_session(self) -> async_sessionmaker[AsyncSession]:
        return async_sessionmaker(self._async_engine, class_=AsyncSession)

    def close(self) -> None:
        self._sync_engine.dispose()
        self._async_engine.sync_engine.dispose()
        logger.debug("SQLite client engines disposed")


class LazySQLiteFactory:
    """Lazy factory for SQLiteClient singleton."""

    _instance: SQLiteClient | None = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, settings: Settings) -> SQLiteClient:
        """Get the singleton instance of SQLiteClient."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = SQLiteClient(
                        local_path=settings.database.local_path,
                        database=settings.database.database,
                    )
        return cls._instance

    @classmethod
    def close_instance(cls) -> None:
        """Close and reset the singleton instance."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.close()
                cls._instance = None
                logger.debug("SQLiteClient instance closed")
