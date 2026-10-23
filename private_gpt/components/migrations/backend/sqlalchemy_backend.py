import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text

from private_gpt.components.migrations.backend.base import MigrationBackend
from private_gpt.components.migrations.models import AppliedMigration

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection, Engine

    from private_gpt.components.migrations.models import Migration

logger = logging.getLogger(__name__)


class SQLAlchemyMigrationBackend(MigrationBackend):
    def __init__(self, engine: "Engine", schema_name: str | None = None) -> None:
        self.engine = engine
        self._schema_name = schema_name
        self._migration_table_name = "schema_migrations"

    def has_migration_table(self) -> bool:
        inspector = sa_inspect(self.engine)
        exists = inspector.has_table(
            self._migration_table_name,
            schema=self._effective_schema(),
        )
        logger.debug(
            "Migration table check: table=%s schema=%s exists=%s",
            self._migration_table_name,
            self._effective_schema(),
            exists,
        )
        return exists

    def applied_migrations(self) -> dict[str, "AppliedMigration"]:
        if not self.has_migration_table():
            return {}

        with self.engine.begin() as conn:
            rows = conn.exec_driver_sql(
                f"SELECT version, description, checksum FROM {self._qualified_table_name()}"
            ).all()

        applied: dict[str, AppliedMigration] = {}
        for row in rows:
            version = str(row[0])
            applied[version] = AppliedMigration(
                version=version,
                description=str(row[1]),
                checksum=row[2],
            )
        logger.debug("Loaded applied migrations: count=%s", len(applied))
        return applied

    def apply(self, migration: "Migration") -> None:
        applied_at = datetime.now(tz=UTC)

        with self.engine.begin() as conn:
            self._ensure_schema(conn)
            migration.up(conn)
            if not sa_inspect(conn).has_table(self._migration_table_name):
                return

            values = {
                "version": migration.version,
                "description": migration.description,
                "migration_date": applied_at,
                "checksum": migration.checksum,
                "applied_at": applied_at,
            }
            existing = conn.execute(
                text(
                    f"SELECT version FROM {self._qualified_table_name()} WHERE version = :version"
                ),
                {"version": migration.version},
            ).first()
            if existing is not None:
                logger.debug(
                    "Migration already tracked, skip insert: version=%s",
                    migration.version,
                )
                return
            conn.execute(
                text(
                    """
                INSERT INTO """
                    + self._qualified_table_name()
                    + """
                (
                    version, description, migration_date, checksum, applied_at
                ) VALUES (
                    :version, :description, :migration_date, :checksum, :applied_at
                )
                """
                ),
                values,
            )
            logger.debug("Migration tracked: version=%s", migration.version)

    def revert(self, migration: "Migration") -> None:
        with self.engine.begin() as conn:
            migration.down(conn)
            if not sa_inspect(conn).has_table(
                self._migration_table_name,
                schema=self._effective_schema(),
            ):
                return
            conn.execute(
                text(
                    f"DELETE FROM {self._qualified_table_name()} WHERE version = :version"
                ),
                {"version": migration.version},
            )
            logger.debug("Migration untracked: version=%s", migration.version)

    def is_applied(self, version: str) -> bool:
        if not self.has_migration_table():
            return False

        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    f"SELECT version FROM {self._qualified_table_name()} WHERE version = :version"
                ),
                {"version": version},
            ).first()
        return row is not None

    def _is_postgres(self) -> bool:
        return self.engine.dialect.name == "postgresql"

    def _effective_schema(self) -> str | None:
        if not self._is_postgres():
            return None
        schema_name = (self._schema_name or "").strip()
        return schema_name or None

    def _ensure_schema(self, conn: "Connection") -> None:
        schema_name = self._effective_schema()
        if schema_name is None:
            return
        conn.exec_driver_sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

    def _qualified_table_name(self) -> str:
        schema_name = self._effective_schema()
        if schema_name is not None:
            return f"{schema_name}.{self._migration_table_name}"
        return self._migration_table_name
