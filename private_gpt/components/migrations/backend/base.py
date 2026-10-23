from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from private_gpt.components.migrations.models import AppliedMigration

if TYPE_CHECKING:
    from private_gpt.components.migrations.models import Migration


class MigrationBackend(ABC):
    @abstractmethod
    def has_migration_table(self) -> bool: ...

    @abstractmethod
    def applied_migrations(self) -> dict[str, AppliedMigration]: ...

    @abstractmethod
    def apply(self, migration: "Migration") -> None: ...

    @abstractmethod
    def revert(self, migration: "Migration") -> None: ...

    @abstractmethod
    def is_applied(self, version: str) -> bool: ...
