import logging
import threading

from private_gpt.components.migrations.backend.base import MigrationBackend
from private_gpt.components.migrations.models import Migration

logger = logging.getLogger(__name__)


class MigrationRunner:
    def __init__(self, backend: MigrationBackend) -> None:
        self._backend = backend
        self._lock = threading.Lock()

    def run_up(self, migrations: list[Migration]) -> None:
        with self._lock:
            has_table = self._backend.has_migration_table()
            applied = self._backend.applied_migrations() if has_table else {}
            logger.info(
                "Starting migrations up: total=%s, has_table=%s, applied=%s",
                len(migrations),
                has_table,
                len(applied),
            )
            for migration in migrations:
                if has_table and migration.version in applied:
                    logger.debug(
                        "Skipping migration version=%s description=%s (already applied)",
                        migration.version,
                        migration.description,
                    )
                    continue
                logger.info(
                    "Applying migration version=%s description=%s",
                    migration.version,
                    migration.description,
                )
                self._backend.apply(migration)
                has_table = has_table or self._backend.has_migration_table()
                applied = self._backend.applied_migrations() if has_table else {}
            logger.info("Migrations up completed: applied=%s", len(applied))

    def run_down(self, migrations: list[Migration], steps: int = 1) -> None:
        if steps <= 0:
            logger.info("Skipping migrations down: steps=%s", steps)
            return

        with self._lock:
            if not self._backend.has_migration_table():
                logger.info("Skipping migrations down: migration table does not exist")
                return

            applied = self._backend.applied_migrations()
            reverted = 0
            logger.info(
                "Starting migrations down: steps=%s, currently_applied=%s",
                steps,
                len(applied),
            )
            for migration in reversed(migrations):
                if reverted >= steps:
                    break
                if migration.version not in applied:
                    continue
                logger.info(
                    "Reverting migration version=%s description=%s",
                    migration.version,
                    migration.description,
                )
                self._backend.revert(migration)
                reverted += 1
            logger.info("Migrations down completed: reverted=%s", reverted)
