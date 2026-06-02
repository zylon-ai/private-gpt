from sqlalchemy import text
from sqlalchemy.engine import Connection

from private_gpt.components.migrations.models import Migration


def _create_skills_tables(connection: Connection) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS skills (
            id VARCHAR(255) PRIMARY KEY,
            collection VARCHAR(255) NOT NULL,
            display_title VARCHAR(255) NOT NULL,
            source VARCHAR(64) NOT NULL,
            loading VARCHAR(16) NOT NULL,
            readonly BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_skills_collection ON skills (collection)
        """,
        """
        CREATE TABLE IF NOT EXISTS skill_versions (
            id VARCHAR(255) PRIMARY KEY,
            skill_id VARCHAR(255) NOT NULL,
            version VARCHAR(255) NOT NULL,
            frontmatter_json JSON NOT NULL,
            storage_prefix VARCHAR(1024) NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            CONSTRAINT uq_skill_version UNIQUE (skill_id, version),
            FOREIGN KEY(skill_id) REFERENCES skills(id) ON DELETE CASCADE
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_skill_versions_skill_id ON skill_versions (skill_id)
        """,
    ]
    for statement in statements:
        connection.execute(text(statement))


def _drop_skills_tables(connection: Connection) -> None:
    statements = [
        "DROP TABLE IF EXISTS skill_versions",
        "DROP TABLE IF EXISTS skills",
    ]
    for statement in statements:
        connection.execute(text(statement))


SKILL_MIGRATIONS: list[Migration] = [
    Migration(
        version="0002",
        description="Create skills and skill_versions tables with current schema",
        up=_create_skills_tables,
        down=_drop_skills_tables,
    ),
]
