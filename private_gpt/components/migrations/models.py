from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.engine import Connection


@dataclass(frozen=True)
class AppliedMigration:
    version: str
    description: str
    checksum: str | None


@dataclass(frozen=True)
class Migration:
    version: str
    description: str
    up: Callable[[Connection], None]
    down: Callable[[Connection], None]
    checksum: str | None = None
