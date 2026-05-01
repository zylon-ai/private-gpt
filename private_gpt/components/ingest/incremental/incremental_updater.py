"""Incremental update pipeline for PrivateGPT.

This is the core module of the proof-of-concept. It orchestrates the entire
incremental update process:

1. Read the modified document and split into semantic chunks
2. Compute hashes for the new chunks
3. Compare with stored hashes to detect changes (using DiffDetector)
4. Re-embed ONLY the changed/added chunks
5. Upsert changed embeddings into the vector store
6. Delete removed chunk embeddings from the vector store
7. Update the chunk hash registry

This avoids the current PrivateGPT behaviour where the entire document is
re-processed on any change, even when only a small portion has been modified.

The pipeline integrates with LlamaIndex's VectorStoreIndex, using:
- index.insert_nodes() for new/modified chunks
- index.delete_ref_doc() or direct node deletion for removed chunks
- StorageContext for persistence

References (from thesis):
- thesis: Methodology -- proof-of-concept design and implementation
- thesis: Embedding management and synchronisation -- upsert operations
- thesis: Problem statement -- inefficiency of full re-ingestion
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from llama_index.core.data_structs import IndexDict
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.indices import VectorStoreIndex, load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from llama_index.core.ingestion import run_transformations
from llama_index.core.schema import (
    Document,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
    TransformComponent,
)
from llama_index.core.storage import StorageContext

from private_gpt.components.ingest.incremental.chunk_hash_store import (
    ChunkHashStore,
    DocumentRecord,
    StoredChunkInfo,
)
from private_gpt.components.ingest.incremental.chunk_hasher import (
    HashedChunk,
    SemanticChunker,
)
from private_gpt.components.ingest.incremental.diff_detector import (
    ChangeType,
    DiffDetector,
)
from private_gpt.components.ingest.ingest_helper import IngestionHelper

logger = logging.getLogger(__name__)


@dataclass
class IncrementalUpdateStats:
    """Statistics about an incremental update operation.

    Used for benchmarking and demonstrating the efficiency gains
    as described in §Methodologie: Experimenten en evaluatie.

    Attributes:
        file_name: The file that was updated.
        total_chunks_old: Number of chunks in the previous version.
        total_chunks_new: Number of chunks in the new version.
        chunks_unchanged: Chunks that didn't need re-embedding.
        chunks_modified: Chunks that were modified and re-embedded.
        chunks_added: New chunks that were embedded and inserted.
        chunks_deleted: Chunks that were removed from the index.
        time_chunking_s: Time spent on chunking and hashing.
        time_diffing_s: Time spent on diff detection.
        time_embedding_s: Time spent computing new embeddings.
        time_indexing_s: Time spent updating the vector store index.
        time_total_s: Total wall-clock time for the incremental update.
        embeddings_computed: Total number of embeddings computed.
        embeddings_skipped: Number of embeddings that were skipped (reused).
    """

    file_name: str = ""
    total_chunks_old: int = 0
    total_chunks_new: int = 0
    chunks_unchanged: int = 0
    chunks_modified: int = 0
    chunks_added: int = 0
    chunks_deleted: int = 0
    time_chunking_s: float = 0.0
    time_diffing_s: float = 0.0
    time_embedding_s: float = 0.0
    time_indexing_s: float = 0.0
    time_total_s: float = 0.0
    embeddings_computed: int = 0
    embeddings_skipped: int = 0

    @property
    def efficiency_ratio(self) -> float:
        """Ratio of skipped embeddings to total chunks.

        A value of 0.8 means 80% of the work was skipped.
        """
        total = self.total_chunks_new + self.chunks_deleted
        if total == 0:
            return 0.0
        return self.embeddings_skipped / max(total, 1)

    def summary(self) -> str:
        """Human-readable summary of the update statistics."""
        return (
            f"Incremental update for '{self.file_name}':\n"
            f"  Chunks: {self.total_chunks_old} old -> {self.total_chunks_new} new\n"
            f"  Unchanged: {self.chunks_unchanged} | Modified: {self.chunks_modified} | "
            f"Added: {self.chunks_added} | Deleted: {self.chunks_deleted}\n"
            f"  Embeddings: {self.embeddings_computed} computed, "
            f"{self.embeddings_skipped} skipped "
            f"({self.efficiency_ratio:.1%} reuse)\n"
            f"  Timing: chunk={self.time_chunking_s:.3f}s, "
            f"diff={self.time_diffing_s:.3f}s, "
            f"embed={self.time_embedding_s:.3f}s, "
            f"index={self.time_indexing_s:.3f}s, "
            f"total={self.time_total_s:.3f}s"
        )


class IncrementalUpdater:
    """Orchestrates incremental document updates within PrivateGPT.

    Instead of re-processing an entire document when it changes, this class:
    1. Chunks the new version using SemanticChunker
    2. Compares chunk hashes with the stored version
    3. Re-embeds only changed chunks
    4. Updates the vector store via upsert-style operations

    Parameters:
        storage_context: LlamaIndex storage context (vector store + doc store).
        embed_model: The embedding model to use.
        transformations: LlamaIndex transformation pipeline.
        persist_dir: Directory for persisting index and hash registry.
        chunker: SemanticChunker instance (or uses defaults).
        diff_detector: DiffDetector instance (or uses defaults).
    """

    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        persist_dir: str | Path,
        chunker: SemanticChunker | None = None,
        diff_detector: DiffDetector | None = None,
        index: BaseIndex[IndexDict] | None = None,
    ) -> None:
        self.storage_context = storage_context
        self.embed_model = embed_model
        self.transformations = transformations
        self.persist_dir = Path(persist_dir)

        self.chunker = chunker or SemanticChunker()
        self.diff_detector = diff_detector or DiffDetector()
        self.hash_store = ChunkHashStore(persist_dir=self.persist_dir)

        # Use the externally-provided index (from BaseIngestComponentWithIndex)
        # or initialize our own.  Accepting an external index avoids loading
        # the same VectorStoreIndex twice when used inside PrivateGPT.
        self._index = index if index is not None else self._initialize_index()

    def _initialize_index(self) -> BaseIndex[IndexDict]:
        """Load existing index or create a new one."""
        try:
            index = load_index_from_storage(
                storage_context=self.storage_context,
                store_nodes_override=True,
                show_progress=True,
                embed_model=self.embed_model,
                transformations=self.transformations,
            )
            logger.info("Loaded existing vector store index")
        except ValueError:
            logger.info("Creating new vector store index for incremental updates")
            index = VectorStoreIndex.from_documents(
                [],
                storage_context=self.storage_context,
                store_nodes_override=True,
                show_progress=True,
                embed_model=self.embed_model,
                transformations=self.transformations,
            )
            index.storage_context.persist(persist_dir=str(self.persist_dir))
        return index

    def _save_index(self) -> None:
        """Persist the index to disk."""
        self._index.storage_context.persist(persist_dir=str(self.persist_dir))

    def _cleanup_prior_ingest(self, file_name: str) -> None:
        """Remove any prior ingest records for this filename.

        Called whenever the incremental pipeline sees a file for the first
        time.  Covers two classes of stale data:

        1. Docstore-tracked entries — scanned via ``get_all_ref_doc_info()``
           and deleted by doc_id.  Catches files that went through the
           standard PrivateGPT pipeline before incremental mode was enabled.

        2. Qdrant orphan points — deleted directly via the Qdrant client
           using a payload filter on ``file_name``.  Catches points whose
           parent ref_doc entry was already removed from the docstore (e.g.
           after a manual data reset or an earlier failed delete).
        """
        # ── 1. Docstore-tracked entries ──────────────────────────────────
        cleaned = 0
        try:
            all_ref_docs = self._index.docstore.get_all_ref_doc_info()
        except Exception as exc:
            logger.debug("Could not read ref_doc_info for duplicate cleanup: %s", exc)
            all_ref_docs = {}

        for ref_doc_id, ref_doc_info in all_ref_docs.items():
            if ref_doc_info is None:
                continue
            meta = getattr(ref_doc_info, "metadata", {}) or {}
            if meta.get("file_name") != file_name:
                continue

            logger.info(
                "Removing prior ingest record for '%s' (doc_id=%s).",
                file_name, ref_doc_id,
            )
            try:
                self._index.storage_context.vector_store.delete(ref_doc_id)
            except Exception as exc:
                logger.debug("vector_store.delete failed for %s: %s", ref_doc_id, exc)
            try:
                self._index.docstore.delete_ref_doc(ref_doc_id, raise_error=False)
            except Exception as exc:
                logger.debug("docstore.delete_ref_doc failed for %s: %s", ref_doc_id, exc)
            cleaned += 1

        # ── 2. Qdrant orphan cleanup by file_name payload filter ──────────
        # Nodes inserted by an old run may have a doc_id that is no longer
        # in the docstore (e.g. after a manual data reset).  The docstore
        # scan above won't find them, but Qdrant still has the points.
        vs = self._index.storage_context.vector_store
        qdrant_client = getattr(vs, "client", None)
        if qdrant_client is not None:
            collection = (
                getattr(vs, "_collection_name", None)
                or getattr(vs, "collection_name", None)
            )
            if collection:
                try:
                    from qdrant_client.models import FieldCondition, Filter, MatchValue
                    qdrant_client.delete(
                        collection_name=collection,
                        points_selector=Filter(
                            must=[
                                FieldCondition(
                                    key="file_name",
                                    match=MatchValue(value=file_name),
                                )
                            ]
                        ),
                    )
                    logger.info(
                        "Deleted all Qdrant orphan points for file_name='%s'.", file_name
                    )
                    cleaned += 1  # count at least one sweep
                except Exception as exc:
                    logger.debug("Qdrant orphan cleanup failed for '%s': %s", file_name, exc)

        if cleaned:
            logger.info(
                "Cleanup complete for '%s' (%d docstore record(s) removed).",
                file_name, cleaned,
            )
            self._save_index()

    def has_file_changed(self, file_name: str, file_data: Path) -> bool:
        """Return False if the file's content hash matches the stored record.

        Used by the watcher callback to silently drop duplicate OS events
        without logging anything.
        """
        try:
            documents = IngestionHelper.transform_file_into_documents(file_name, file_data)
            full_text = "\n\n".join(doc.text for doc in documents)
            file_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
            existing = self.hash_store.get_document_by_filename(file_name)
            return not (existing and existing.file_hash == file_hash)
        except Exception:
            return True  # on error, let the full pipeline decide

    def ingest_file(
        self, file_name: str, file_data: Path
    ) -> tuple[IncrementalUpdateStats, list[Document]]:
        """Ingest a file using incremental updates.

        If the file has been ingested before, only changed chunks are
        re-embedded and updated. If it's a new file, all chunks are processed.

        Args:
            file_name: The display name of the file.
            file_data: Path to the file on disk.

        Returns:
            A tuple of (stats, documents) where *documents* are the
            LlamaIndex Document objects parsed from the file.
        """
        stats = IncrementalUpdateStats(file_name=file_name)
        total_start = time.perf_counter()

        # Step 1: Read the file content
        logger.info("Starting incremental ingest for file=%s", file_name)
        documents = IngestionHelper.transform_file_into_documents(file_name, file_data)

        # Combine all document texts (a file may produce multiple Documents)
        full_text = "\n\n".join(doc.text for doc in documents)
        file_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()

        # Check if the file has been processed before
        existing_record = self.hash_store.get_document_by_filename(file_name)

        if existing_record and existing_record.file_hash == file_hash:
            # File hasn't changed at all – skip entirely
            logger.info("File %s unchanged (same file hash). Skipping.", file_name)
            stats.time_total_s = time.perf_counter() - total_start
            stats.total_chunks_old = len(existing_record.chunks)
            stats.total_chunks_new = len(existing_record.chunks)
            stats.chunks_unchanged = len(existing_record.chunks)
            stats.embeddings_skipped = len(existing_record.chunks)
            return stats, documents

        # Step 2: Chunk the new version
        chunk_start = time.perf_counter()
        metadata = {"file_name": file_name}
        new_chunks = self.chunker.chunk_text(full_text, metadata=metadata)
        stats.time_chunking_s = time.perf_counter() - chunk_start
        stats.total_chunks_new = len(new_chunks)

        if existing_record is None:
            # New file – full ingestion (but through our chunking pipeline).
            # Clean up any prior standard-pipeline records first to prevent
            # duplicate embeddings (same filename, different doc_id scheme).
            self._cleanup_prior_ingest(file_name)
            logger.info(
                "New file %s: performing full ingestion of %d chunks",
                file_name,
                len(new_chunks),
            )
            return self._full_ingest(
                file_name, file_hash, documents, new_chunks, stats, total_start
            )

        # Step 3: Build old chunks from stored record
        diff_start = time.perf_counter()
        old_chunks = [
            HashedChunk(
                chunk_index=stored.chunk_index,
                text=stored.full_text or stored.text_preview,
                content_hash=stored.content_hash,
            )
            for stored in existing_record.chunks
        ]
        stats.total_chunks_old = len(old_chunks)

        # Step 4: Detect changes
        changes = self.diff_detector.detect_changes(old_chunks, new_chunks)
        stats.time_diffing_s = time.perf_counter() - diff_start

        # Categorise changes
        added_chunks = [c for c in changes if c.change_type == ChangeType.ADDED]
        modified_chunks = [c for c in changes if c.change_type == ChangeType.MODIFIED]
        deleted_chunks = [c for c in changes if c.change_type == ChangeType.DELETED]
        unchanged_chunks = [c for c in changes if c.change_type == ChangeType.UNCHANGED]

        stats.chunks_added = len(added_chunks)
        stats.chunks_modified = len(modified_chunks)
        stats.chunks_deleted = len(deleted_chunks)
        stats.chunks_unchanged = len(unchanged_chunks)
        stats.embeddings_skipped = len(unchanged_chunks)

        # Step 5: Embed ALL new chunks BEFORE touching the live index.
        # Computing embeddings takes several seconds.  If we deleted old nodes
        # first, any concurrent RAG query would return empty results during the
        # embedding window.  By embedding first and then doing a fast
        # delete+insert, the gap where Qdrant has no content for this file is
        # reduced to milliseconds.
        stats.embeddings_skipped = 0
        embed_start = time.perf_counter()
        chunks_to_embed: list[HashedChunk] = new_chunks  # re-embed everything

        new_node_map: dict[int, str] = {}  # chunk_index -> node_id
        embedded_nodes: list = []

        if chunks_to_embed:
            text_nodes = self._create_text_nodes(
                chunks_to_embed, file_name, existing_record.doc_id
            )
            embedded_nodes = run_transformations(
                text_nodes,
                self.transformations,
                show_progress=True,
            )
            stats.embeddings_computed = len(embedded_nodes)

        stats.time_embedding_s = time.perf_counter() - embed_start

        # Step 6: Wipe ALL prior nodes for this document, then insert the
        # freshly-embedded nodes.  Both operations are fast (no model calls),
        # keeping the empty-Qdrant window as small as possible.
        #
        # Direct backend access bypasses VectorStoreIndex.IndexDict which
        # raises KeyError for nodes inserted by a different VectorStoreIndex
        # instance (e.g. IngestService vs. IncrementalIngestService singletons).
        index_start = time.perf_counter()
        try:
            self._index.storage_context.vector_store.delete(existing_record.doc_id)
            logger.debug(
                "Wiped all vectors for doc_id=%s (file=%s)",
                existing_record.doc_id, file_name,
            )
        except Exception as exc:
            logger.debug(
                "vector_store.delete failed for %s (%s), continuing",
                existing_record.doc_id, exc,
            )
        try:
            self._index.docstore.delete_ref_doc(
                existing_record.doc_id, raise_error=False
            )
        except Exception as exc:
            logger.debug(
                "docstore.delete_ref_doc failed for %s (%s)",
                existing_record.doc_id, exc,
            )

        if embedded_nodes:
            self._index.insert_nodes(embedded_nodes, show_progress=True)
            for node, chunk in zip(embedded_nodes, chunks_to_embed):
                new_node_map[chunk.chunk_index] = node.node_id

        stats.time_indexing_s = time.perf_counter() - index_start

        # Step 8: Update the chunk hash registry.
        # All non-deleted chunks were re-embedded and inserted above, so every
        # surviving chunk has a fresh node_id in new_node_map.
        stored_chunks: list[StoredChunkInfo] = []

        for change in changes:
            if change.change_type == ChangeType.DELETED or change.new_chunk is None:
                continue  # Deleted chunks are removed from the index entirely
            chunk = change.new_chunk
            stored_chunks.append(
                StoredChunkInfo(
                    chunk_index=chunk.chunk_index,
                    content_hash=chunk.content_hash,
                    node_id=new_node_map.get(chunk.chunk_index, ""),
                    text_preview=chunk.text[:100],
                    full_text=chunk.text,
                )
            )

        # Sort by chunk index
        stored_chunks.sort(key=lambda c: c.chunk_index)

        updated_record = DocumentRecord(
            doc_id=existing_record.doc_id,
            file_name=file_name,
            file_hash=file_hash,
            chunks=stored_chunks,
            version=existing_record.version + 1,
        )
        self.hash_store.upsert_document(updated_record)

        # Persist everything
        self._save_index()

        stats.time_total_s = time.perf_counter() - total_start
        logger.info(stats.summary())
        return stats, documents

    def _full_ingest(
        self,
        file_name: str,
        file_hash: str,
        documents: list[Document],
        new_chunks: list[HashedChunk],
        stats: IncrementalUpdateStats,
        total_start: float,
    ) -> tuple[IncrementalUpdateStats, list[Document]]:
        """Perform full ingestion for a new file.

        Even for new files, we use our semantic chunking and store the
        hashes so that future updates can be incremental.
        """
        # Use the first document's doc_id as the canonical ID
        doc_id = documents[0].doc_id if documents else file_name

        # Create TextNodes from our semantic chunks
        text_nodes = self._create_text_nodes(new_chunks, file_name, doc_id)

        # Run transformations (includes embedding)
        embed_start = time.perf_counter()
        embedded_nodes = run_transformations(
            text_nodes,
            self.transformations,
            show_progress=True,
        )
        stats.time_embedding_s = time.perf_counter() - embed_start
        stats.embeddings_computed = len(embedded_nodes)

        # Insert into index
        index_start = time.perf_counter()
        self._index.insert_nodes(embedded_nodes, show_progress=True)

        # Set document hash in docstore for LlamaIndex compatibility
        for document in documents:
            self._index.docstore.set_document_hash(document.doc_id, document.hash)

        stats.time_indexing_s = time.perf_counter() - index_start

        # Store chunk hashes for future incremental updates
        stored_chunks = []
        for node, chunk in zip(embedded_nodes, new_chunks):
            stored_chunks.append(
                StoredChunkInfo(
                    chunk_index=chunk.chunk_index,
                    content_hash=chunk.content_hash,
                    node_id=node.node_id,
                    text_preview=chunk.text[:100],
                    full_text=chunk.text,
                )
            )

        record = DocumentRecord(
            doc_id=doc_id,
            file_name=file_name,
            file_hash=file_hash,
            chunks=stored_chunks,
            version=1,
        )
        self.hash_store.upsert_document(record)

        # Persist everything
        self._save_index()

        stats.chunks_added = len(new_chunks)
        stats.time_total_s = time.perf_counter() - total_start
        logger.info(stats.summary())
        return stats, documents

    def delete_file(self, file_name: str) -> bool:
        """Remove all chunks for a file from the index and hash store.

        Args:
            file_name: The file name to remove.

        Returns:
            True if the file was found and deleted, False otherwise.
        """
        record = self.hash_store.get_document_by_filename(file_name)
        if record is None:
            logger.warning("File %s not found in hash store", file_name)
            return False

        # Delete all vectors for this document from the vector store.
        # Use the direct backend rather than VectorStoreIndex.delete_ref_doc /
        # delete_nodes: those methods update IndexDict in memory, which raises
        # KeyError when the nodes were inserted by a different VectorStoreIndex
        # instance (e.g. IngestService vs. IncrementalIngestService).
        try:
            self._index.storage_context.vector_store.delete(record.doc_id)
            logger.info(
                "Deleted all vectors for file=%s (doc_id=%s)", file_name, record.doc_id
            )
        except Exception as exc:
            logger.warning(
                "vector_store.delete failed for %s (%s)", file_name, exc
            )

        # Remove the ref_doc entry from the docstore so the file no longer
        # appears in list_ingested().
        try:
            self._index.docstore.delete_ref_doc(record.doc_id, raise_error=False)
        except Exception as exc:
            logger.warning(
                "docstore.delete_ref_doc failed for %s (%s)", file_name, exc
            )

        # Remove from hash store
        self.hash_store.delete_document(record.doc_id)
        self._save_index()
        return True

    def _create_text_nodes(
        self,
        chunks: list[HashedChunk],
        file_name: str,
        doc_id: str,
    ) -> list[TextNode]:
        """Create LlamaIndex TextNode objects from HashedChunks.

        Each TextNode carries metadata for:
        - file_name: Source file
        - doc_id: Parent document ID (for ref_doc tracking)
        - chunk_index: Position in document
        - content_hash: SHA-256 hash (for future change detection)
        """
        nodes = []
        for chunk in chunks:
            node = TextNode(
                text=chunk.text,
                metadata={
                    "file_name": file_name,
                    "doc_id": doc_id,
                    "chunk_index": chunk.chunk_index,
                    "content_hash": chunk.content_hash,
                },
                excluded_embed_metadata_keys=["doc_id", "chunk_index", "content_hash"],
                excluded_llm_metadata_keys=["doc_id", "chunk_index", "content_hash"],
            )
            # Set the ref_doc_id so LlamaIndex can track the parent document.
            # This makes docstore.get_all_ref_doc_info() return entries for
            # incrementally-ingested files, which is required for list_ingested()
            # and the UI file list.
            node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                node_id=doc_id,
                metadata={"file_name": file_name},
            )
            nodes.append(node)
        return nodes

    def get_stats_for_file(self, file_name: str) -> dict:
        """Get information about how a file is stored in the hash registry."""
        record = self.hash_store.get_document_by_filename(file_name)
        if record is None:
            return {"status": "not_found"}
        return {
            "status": "found",
            "doc_id": record.doc_id,
            "file_name": record.file_name,
            "version": record.version,
            "num_chunks": len(record.chunks),
            "file_hash": record.file_hash,
        }
