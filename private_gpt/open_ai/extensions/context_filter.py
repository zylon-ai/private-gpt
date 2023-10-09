from dataclasses import dataclass


@dataclass
class ContextFilter:
    docs_ids: list[str] | None = None
