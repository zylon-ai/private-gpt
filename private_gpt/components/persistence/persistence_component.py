import logging
import threading
from typing import Any

from injector import inject, singleton

from private_gpt.components.migrations.backend.base import MigrationBackend
from private_gpt.components.migrations.runner import MigrationRunner
from private_gpt.components.persistence.migrations import MIGRATIONS
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class PersistenceComponent:
    @inject
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.RLock()
        self._clients: dict[str, Any] = {}
        self._migrations_applied = False

    def get_client(
        self,
        store: str,
    ) -> Any | None:
        if store in self._clients:
            return self._clients[store]

        client: Any | None = None
        match store:
            case "postgres":
                from private_gpt.components.postgres.postgres_client import (
                    LazyPostgresFactory,
                )

                client = LazyPostgresFactory.get_instance(self._settings)
            case "sqlite":
                from private_gpt.components.sqlite.sqlite_client import (
                    LazySQLiteFactory,
                )

                local_path = self._settings.database.local_path
                if local_path is None:
                    raise ValueError("local_path is required for sqlite client")
                client = LazySQLiteFactory.get_instance(self._settings)

        self._clients[store] = client
        return client

    def _get_migration_backend(
        self,
        store: str,
        client: Any,
    ) -> MigrationBackend:
        match store:
            case "postgres" | "sqlite":
                from private_gpt.components.migrations.backend.sqlalchemy_backend import (
                    SQLAlchemyMigrationBackend,
                )

                engine = client.sync_session.kw.get("bind")
                if engine is None:
                    raise ValueError(
                        f"Cannot resolve SQLAlchemy engine for migration store '{store}'"
                    )
                return SQLAlchemyMigrationBackend(
                    engine=engine,
                    schema_name=self._settings.database.schema_name,
                )
            case _:
                raise ValueError(f"Unsupported store type: {store}")

    def apply_migrations(self) -> None:
        with self._lock:
            if self._migrations_applied:
                logger.info("Migrations already applied in this process; skipping")
                return

            store = self._settings.database.provider
            schema = self._settings.database.schema_name
            logger.info(
                "Applying migrations with provider=%s schema=%s total=%s",
                store,
                schema,
                len(MIGRATIONS),
            )
            client = self.get_client(store)
            if client is None:
                raise ValueError(
                    f"Migration client is not available for store '{store}'"
                )

            migration_backend: MigrationBackend = self._get_migration_backend(
                store=store,
                client=client,
            )
            runner = MigrationRunner(migration_backend)
            runner.run_up(migrations=list(MIGRATIONS))
            self._migrations_applied = True
            logger.info("Migrations applied successfully")

    # Testing use: This method will revert all migrations
    def revert_migrations(self) -> None:
        with self._lock:
            store = self._settings.database.provider
            schema = self._settings.database.schema_name
            logger.info(
                "Reverting migrations with provider=%s schema=%s total=%s",
                store,
                schema,
                len(MIGRATIONS),
            )
            client = self.get_client(store)
            if client is None:
                raise ValueError(
                    f"Migration client is not available for store '{store}'"
                )

            migration_backend: MigrationBackend = self._get_migration_backend(
                store=store,
                client=client,
            )
            runner = MigrationRunner(migration_backend)
            runner.run_down(migrations=list(MIGRATIONS))
            logger.info("Migrations reverted")
