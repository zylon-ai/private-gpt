from dataclasses import dataclass


@dataclass
class ContextFiles:
    collection: list[str] | None = None
    ids: list[str] | None = None
