from dataclasses import dataclass


@dataclass
class ContextFiles:
    collections: list[str] | None = None
    ids: list[str] | None = None
