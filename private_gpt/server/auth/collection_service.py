"""Maps named collections to document IDs for query scoping."""

import logging

from injector import inject, singleton

from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.ingest.ingest_service import IngestService

logger = logging.getLogger(__name__)


@singleton
class CollectionService:
    @inject
    def __init__(self, ingest_service: IngestService) -> None:
        self._ingest_service = ingest_service

    def get_collection_doc_ids(self, collection_name: str) -> list[str]:
        """Return all doc_ids whose metadata matches *collection_name*."""
        doc_ids = []
        for doc in self._ingest_service.list_ingested():
            if doc.doc_metadata and doc.doc_metadata.get("collection_name") == collection_name:
                doc_ids.append(doc.doc_id)
        return doc_ids

    def build_context_filter(self, collection_name: str) -> ContextFilter | None:
        """Return a ContextFilter scoped to *collection_name*, or None if empty."""
        doc_ids = self.get_collection_doc_ids(collection_name)
        if not doc_ids:
            # Return a filter with an impossible doc_id so zero results come back
            return ContextFilter(docs_ids=["__no_docs__"])
        return ContextFilter(docs_ids=doc_ids)

    def list_known_collections(self) -> list[str]:
        """Return distinct collection names present in ingested documents."""
        names: set[str] = set()
        for doc in self._ingest_service.list_ingested():
            if doc.doc_metadata:
                col = doc.doc_metadata.get("collection_name")
                if col:
                    names.add(col)
        return sorted(names)
