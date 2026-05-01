"""Persistent storage for chunk hashes.

This module maintains a JSON-based registry that maps each document to its
list of chunk hashes. This enables efficient change detection: when a document
is re-ingested, the new chunk hashes are compared against the stored hashes
to determine which chunks have been added, modified, or deleted.

The registry is stored alongside PrivateGPT's other persistent data in the
local_data directory.

References (from thesis):
- Hash-based change detection (LangChain Sync Vector Stores, 2023)
- Content-based change detection for incremental updates (§Risico's en beperkingen)
"""

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StoredChunkInfo:
    """Information stored for each chunk in the registry.

    Attributes:
        chunk_index: The position of the chunk in the document.
        content_hash: SHA-256 hash of the chunk content.
        node_id: The LlamaIndex node ID for this chunk (used for updates/deletes).
        text_preview: First 100 characters of the chunk (for debugging).
        full_text: Complete chunk text, stored so that the DiffDetector's
                   SequenceMatcher can compute meaningful similarity ratios
                   when matching modified chunks (instead of comparing against
                   a truncated 100-char preview).
    """

    chunk_index: int
    content_hash: str
    node_id: str = ""
    text_preview: str = ""
    full_text: str = ""


@dataclass
class DocumentRecord:
    """Record of a processed document in the hash store.

    Attributes:
        doc_id: Unique document identifier (from LlamaIndex).
        file_name: Original file name.
        file_hash: Hash of the entire file content (for quick skip detection).
        chunks: List of chunk info for this document.
        version: Monotonically increasing version number.
    """

    doc_id: str
    file_name: str
    file_hash: str = ""
    chunks: list[StoredChunkInfo] = field(default_factory=list)
    version: int = 1


class ChunkHashStore:
    """Persistent store for document chunk hashes.

    Maintains a JSON file mapping document IDs to their chunk hashes.
    Thread-safe via a reentrant lock.

    The store file is created in the provided `persist_dir` as
    `chunk_hash_registry.json`.

    Parameters:
        persist_dir: Directory where the hash registry file is stored.
    """

    REGISTRY_FILENAME = "chunk_hash_registry.json"

    def __init__(self, persist_dir: str | Path) -> None:
        self._persist_dir = Path(persist_dir)
        self._registry_path = self._persist_dir / self.REGISTRY_FILENAME
        self._lock = threading.RLock()
        self._documents: dict[str, DocumentRecord] = {}
        self._last_load_mtime: float = 0.0
        self._load()

    def _load(self) -> None:
        """Load the registry from disk, if it exists."""
        if self._registry_path.exists():
            try:
                with open(self._registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for doc_id, doc_data in data.items():
                    chunks = [
                        StoredChunkInfo(**chunk) for chunk in doc_data.get("chunks", [])
                    ]
                    self._documents[doc_id] = DocumentRecord(
                        doc_id=doc_data["doc_id"],
                        file_name=doc_data["file_name"],
                        file_hash=doc_data.get("file_hash", ""),
                        chunks=chunks,
                        version=doc_data.get("version", 1),
                    )
                self._last_load_mtime = self._registry_path.stat().st_mtime
                logger.info(
                    "Loaded chunk hash registry with %d documents from %s",
                    len(self._documents),
                    self._registry_path,
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    "Failed to load chunk hash registry: %s. Starting fresh.", e
                )
                self._documents = {}
                self._last_load_mtime = 0.0
        else:
            self._last_load_mtime = 0.0
            logger.info(
                "No chunk hash registry found at %s. Starting fresh.",
                self._registry_path,
            )

    def _reload_if_stale(self) -> None:
        """Reload from disk when another instance has written a newer version.

        Two ``ChunkHashStore`` instances (one in ``IncrementalIngestComponent``,
        one in ``IncrementalIngestService``) share the same JSON file.  Without
        this check, the instance that last wrote the file would see up-to-date
        data while the other sees a stale in-memory snapshot — leading to
        phantom records (duplicates) or missing records (missed cleanup).

        A single ``os.stat()`` call per read is negligible overhead.
        """
        try:
            if (
                self._registry_path.exists()
                and self._registry_path.stat().st_mtime > self._last_load_mtime
            ):
                logger.debug(
                    "Chunk hash registry changed on disk — reloading from %s",
                    self._registry_path,
                )
                self._load()
        except Exception:
            pass  # Best-effort; a stale read is acceptable as a fallback

    def _save(self) -> None:
        """Persist the registry to disk."""
        os.makedirs(self._persist_dir, exist_ok=True)
        data = {}
        for doc_id, record in self._documents.items():
            data[doc_id] = {
                "doc_id": record.doc_id,
                "file_name": record.file_name,
                "file_hash": record.file_hash,
                "version": record.version,
                "chunks": [asdict(chunk) for chunk in record.chunks],
            }
        with open(self._registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug("Saved chunk hash registry to %s", self._registry_path)

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        """Get the stored record for a document, or None if not found."""
        with self._lock:
            self._reload_if_stale()
            return self._documents.get(doc_id)

    def get_document_by_filename(self, file_name: str) -> DocumentRecord | None:
        """Look up a document record by file name."""
        with self._lock:
            self._reload_if_stale()
            for record in self._documents.values():
                if record.file_name == file_name:
                    return record
            return None

    def upsert_document(self, record: DocumentRecord) -> None:
        """Insert or update a document record and persist."""
        with self._lock:
            self._documents[record.doc_id] = record
            self._save()

    def delete_document(self, doc_id: str) -> None:
        """Remove a document record and persist."""
        with self._lock:
            if doc_id in self._documents:
                del self._documents[doc_id]
                self._save()
                logger.info("Deleted document %s from chunk hash registry", doc_id)

    def list_documents(self) -> list[DocumentRecord]:
        """Return all stored document records."""
        with self._lock:
            return list(self._documents.values())

    def get_chunk_hashes(self, doc_id: str) -> dict[int, str]:
        """Return a mapping of chunk_index -> content_hash for a document."""
        with self._lock:
            record = self._documents.get(doc_id)
            if record is None:
                return {}
            return {chunk.chunk_index: chunk.content_hash for chunk in record.chunks}

    def get_chunk_node_ids(self, doc_id: str) -> dict[int, str]:
        """Return a mapping of chunk_index -> node_id for a document."""
        with self._lock:
            record = self._documents.get(doc_id)
            if record is None:
                return {}
            return {chunk.chunk_index: chunk.node_id for chunk in record.chunks}
