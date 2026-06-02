from sqlalchemy import text
from sqlalchemy.engine import Connection

from private_gpt.components.migrations.models import Migration
from private_gpt.components.skills.migrations import SKILL_MIGRATIONS


def _create_schema_migrations_table(connection: Connection) -> None:
    connection.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(255) PRIMARY KEY,
                description VARCHAR(1024) NOT NULL,
                migration_date TIMESTAMP NOT NULL,
                checksum VARCHAR(128),
                applied_at TIMESTAMP NOT NULL
            )
            """
        )
    )


def _drop_schema_migrations_table(connection: Connection) -> None:
    connection.execute(text("DROP TABLE IF EXISTS schema_migrations"))


INITIAL_MIGRATIONS: list[Migration] = [
    Migration(
        version="0001",
        description="Initial migration: create schema_migrations table",
        up=_create_schema_migrations_table,
        down=_drop_schema_migrations_table,
    )
]

MIGRATIONS = [
    # Initial migration
    *INITIAL_MIGRATIONS,
    # Skill Models
    *SKILL_MIGRATIONS,
]
