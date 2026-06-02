import logging
import threading
from importlib.util import find_spec

from private_gpt.utils.dependencies import format_missing_dependency_message

if find_spec("psycopg2") is None or find_spec("sqlalchemy") is None:
    raise ImportError(
        format_missing_dependency_message(
            "Postgres client",
            extras=("database-postgres", "nodestore-postgres"),
        )
    )

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PostgresClient:
    def __init__(self, settings: Settings):
        self._settings = settings

        self.sync_session = self.new_sync_session()
        self.async_session = self.new_async_session()

    def new_sync_session(self) -> sessionmaker[Session]:
        schema_name = self._settings.database.schema_name
        engine = create_engine(
            f"postgresql+psycopg2://{self._settings.database.url_without_protocol}",
            connect_args={"options": f"-csearch_path={schema_name}"},
            execution_options={"schema_translate_map": {"app": schema_name}},
        )
        return sessionmaker(engine)

    def new_async_session(self) -> sessionmaker[Session]:
        schema_name = self._settings.database.schema_name
        engine = create_async_engine(
            f"postgresql+asyncpg://{self._settings.database.url_without_protocol}",
            connect_args={"server_settings": {"search_path": schema_name}},
            execution_options={"schema_translate_map": {"app": schema_name}},
        )
        return sessionmaker(engine, class_=AsyncSession)  # type: ignore

    def close(self) -> None:
        """Close the sync session when the client is deleted."""
        if not self:
            return
        if not self.sync_session and not self.async_session:
            return

        logger.debug("Closing Postgres client sessions")
        if hasattr(self, "sync_session") and self.sync_session:
            self.sync_session.close_all()
            del self.sync_session
        if hasattr(self, "async_session") and self.async_session:
            self.async_session.close_all()
            del self.async_session

    def __del__(self) -> None:
        """Ensure the sync session is closed when the client is deleted."""
        self.close()
        del self


class LazyPostgresFactory:
    """Lazy factory for PostgresClient singleton."""

    _instance: PostgresClient | None = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, settings: Settings) -> PostgresClient:
        """Get the singleton instance of PostgresClient."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = PostgresClient(settings)
        return cls._instance

    @classmethod
    def close_instance(cls) -> None:
        """Close and reset the singleton instance."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.close()
                cls._instance = None
                logger.debug("PostgresClient instance closed")
