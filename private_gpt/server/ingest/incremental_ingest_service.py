"""Incremental ingest service for PrivateGPT.

This service wraps the IncrementalUpdater and IncrementalIngestWatcher
to provide a high-level API for incremental document processing within
PrivateGPT's dependency injection framework.

It follows the same patterns as the existing IngestService but routes
through the incremental pipeline when documents have been seen before.

The service can be used:
1. Programmatically via `incremental_ingest_file()`
2. Automatically via `start_watcher()` for continuous monitoring

References (from thesis):
- thesis: Methodology -- integration into PrivateGPT
- thesis: Research goal -- extension with file-watcher + update pipeline
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, BinaryIO

from injector import inject, singleton
from llama_index.core.storage import StorageContext

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.ingest.incremental.chunk_hasher import SemanticChunker
from private_gpt.components.ingest.incremental.diff_detector import DiffDetector
from private_gpt.components.ingest.incremental.incremental_updater import (
    IncrementalUpdater,
    IncrementalUpdateStats,
)
from private_gpt.components.ingest.incremental.incremental_watcher import (
    IncrementalIngestWatcher,
)
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.paths import local_data_path
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class IncrementalIngestService:
    """Service for incremental document ingestion.

    This service provides the same interface as IngestService but with
    incremental update capabilities. When a document that has been previously
    ingested is modified, only the changed chunks are re-embedded and
    updated in the vector store.

    Usage:
        # Direct injection (following PrivateGPT patterns)
        service = IncrementalIngestService(
            llm_component, vector_store_component,
            embedding_component, node_store_component
        )

        # Ingest a file (incremental if previously seen)
        stats = service.incremental_ingest_file(
            "report.pdf", Path("/path/to/report.pdf"),
        )

        # Start continuous watching
        service.start_watcher(Path("/path/to/documents"))
    """

    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
        settings_service: Settings,
    ) -> None:
        # Read incremental config from settings (falls back to defaults)
        inc_settings = settings_service.incremental
        min_chunk_size = inc_settings.min_chunk_size
        max_chunk_size = inc_settings.max_chunk_size
        similarity_threshold = inc_settings.similarity_threshold

        self.settings = inc_settings
        self.llm_service = llm_component
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )

        self.embedding_model = embedding_component.embedding_model

        # Initialise incremental components
        self.chunker = SemanticChunker(
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )
        self.diff_detector = DiffDetector(
            similarity_threshold=similarity_threshold,
        )

        self.updater = IncrementalUpdater(
            storage_context=self.storage_context,
            embed_model=self.embedding_model,
            # Only the embedding transform — no node parsers.
            # SentenceWindowNodeParser would create multiple Qdrant nodes per
            # semantic chunk, breaking the 1:1 chunk↔node_id mapping that the
            # hash store relies on for targeted deletion during updates.
            transformations=[self.embedding_model],
            persist_dir=str(local_data_path),
            chunker=self.chunker,
            diff_detector=self.diff_detector,
        )

        self._watcher = IncrementalIngestWatcher(
            debounce_seconds=inc_settings.debounce_seconds,
        )
        self._persist_path = local_data_path / "watched_files.json"
        self._update_history: list[IncrementalUpdateStats] = []

    def incremental_ingest_file(
        self, file_name: str, file_data: Path
    ) -> IncrementalUpdateStats:
        """Ingest a file with incremental update support.

        If the file has been ingested before, only changed chunks are
        re-embedded. If it's new, a full ingestion is performed.

        Args:
            file_name: Display name of the file.
            file_data: Path to the file on disk.

        Returns:
            Statistics about the update operation.
        """
        logger.info("Incremental ingest for file_name=%s", file_name)
        stats, _documents = self.updater.ingest_file(file_name, file_data)
        self._update_history.append(stats)
        return stats

    def incremental_ingest_text(
        self, file_name: str, text: str
    ) -> IncrementalUpdateStats:
        """Ingest text data with incremental update support."""
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".txt", mode="w", encoding="utf-8"
        )
        tmp_path = Path(tmp.name)
        try:
            tmp.write(text)
            tmp.close()  # Close before ingesting (Windows file locking)
            return self.incremental_ingest_file(file_name, tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def incremental_ingest_bin_data(
        self, file_name: str, raw_file_data: BinaryIO
    ) -> IncrementalUpdateStats:
        """Ingest binary data with incremental update support."""
        file_data = raw_file_data.read()
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp_path = Path(tmp.name)
        try:
            tmp.write(file_data)
            tmp.close()  # Close before ingesting (Windows file locking)
            return self.incremental_ingest_file(file_name, tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def delete_file(self, file_name: str) -> bool:
        """Delete a file and all its chunks from the index."""
        return self.updater.delete_file(file_name)

    def get_file_info(self, file_name: str) -> dict[str, Any]:
        """Get information about how a file is stored."""
        return self.updater.get_stats_for_file(file_name)

    def get_update_history(self) -> list[dict[str, Any]]:
        """Get the history of all incremental updates in this session."""
        return [
            {
                "file_name": s.file_name,
                "total_chunks_old": s.total_chunks_old,
                "total_chunks_new": s.total_chunks_new,
                "chunks_unchanged": s.chunks_unchanged,
                "chunks_modified": s.chunks_modified,
                "chunks_added": s.chunks_added,
                "chunks_deleted": s.chunks_deleted,
                "embeddings_computed": s.embeddings_computed,
                "embeddings_skipped": s.embeddings_skipped,
                "efficiency_ratio": s.efficiency_ratio,
                "time_total_s": s.time_total_s,
            }
            for s in self._update_history
        ]

    # ─── Watcher Integration ─────────────────────────────────────────

    def touch_debounce(self, file_path: Path) -> None:
        """Pre-arm the watcher debounce for a file BEFORE modifying it on disk.

        Call this before shutil.copy2 (or any write) to prevent the watcher
        from treating the upload-triggered file change as an independent save.
        """
        self._watcher.touch_debounce(file_path)

    def ingest_file_from_path(self, file_path: Path) -> IncrementalUpdateStats:
        """Ingest a file from its path on disk and register it for watching.

        This is the primary entry-point for the file-watcher workflow:
        the file is ingested incrementally (or fully on first sight) and
        then watched so that any future saves automatically trigger a
        re-ingestion of only the changed chunks.

        Args:
            file_path: Absolute path to the file on the local filesystem.

        Returns:
            Statistics about the ingestion operation.
        """
        file_path = file_path.resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        stats = self.incremental_ingest_file(file_path.name, file_path)
        self._register_file_for_watching(file_path)
        return stats

    def _register_file_for_watching(self, file_path: Path) -> None:
        """Add a file to the watcher and persist the list."""
        file_path = file_path.resolve()

        def on_modified(path: Path) -> None:
            if not self.updater.has_file_changed(path.name, path):
                return
            logger.info(
                "File change detected: %s — triggering incremental update", path.name
            )
            try:
                stats = self.incremental_ingest_file(path.name, path)
                logger.info(stats.summary())
            except Exception as e:
                logger.error("Failed to update %s: %s", path.name, e)

        def on_deleted(path: Path) -> None:
            logger.info("File deleted: %s — removing from index", path.name)
            try:
                self.delete_file(path.name)
            except Exception as e:
                logger.error("Failed to delete %s from index: %s", path.name, e)

        self._watcher.add_file_watch(
            file_path, on_modified=on_modified, on_deleted=on_deleted
        )
        self._persist_watched_files()

    def unwatch_file(self, file_path: Path) -> bool:
        """Stop watching a file. Returns True if it was registered."""
        removed = self._watcher.remove_file_watch(file_path.resolve())
        if removed:
            self._persist_watched_files()
        return removed

    def unwatch_all_files(self) -> int:
        """Unregister every watched file. Returns the number removed."""
        paths = list(self._watcher.registered_file_paths)
        removed = 0
        for p in paths:
            if self._watcher.remove_file_watch(p):
                removed += 1
        self._persist_watched_files()
        logger.info("Unwatched all %d registered file(s).", removed)
        return removed

    def start_watching_background(self) -> None:
        """Start the observer thread and reload all persisted watched files."""
        self._load_watched_files()
        self._watcher.start_background()
        logger.info(
            "Incremental file watcher started — %d file(s) registered.",
            len(self._watcher.registered_file_paths),
        )

    def stop_watching(self) -> None:
        """Stop the observer thread. Registered file list is preserved."""
        self._watcher.stop()

    @property
    def is_watcher_running(self) -> bool:
        """True if the observer thread is currently active."""
        return self._watcher.is_running

    @property
    def watched_file_paths(self) -> list[Path]:
        """All files currently registered for watching."""
        return self._watcher.registered_file_paths

    # ─── Persistence ─────────────────────────────────────────────────

    def _persist_watched_files(self) -> None:
        """Save the current watched-file list to disk."""
        paths = [str(p) for p in self._watcher.registered_file_paths]
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(json.dumps(paths, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Could not persist watched files: %s", e)

    def _load_watched_files(self) -> None:
        """Load the persisted watched-file list and register each file."""
        if not self._persist_path.exists():
            return
        try:
            paths: list[str] = json.loads(
                self._persist_path.read_text(encoding="utf-8")
            )
        except Exception as e:
            logger.warning("Could not load watched files: %s", e)
            return

        loaded = 0
        for path_str in paths:
            p = Path(path_str)
            if p.exists():
                self._register_file_for_watching(p)
                loaded += 1
            else:
                logger.warning("Watched file no longer exists, skipping: %s", path_str)
        logger.info("Loaded %d/%d previously watched file(s).", loaded, len(paths))
