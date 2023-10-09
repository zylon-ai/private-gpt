from dataclasses import dataclass


@dataclass
class ContextDocs:
    # Could be "all", a list of doc_ids or None
    docs_ids: str | list[str] | None = None
