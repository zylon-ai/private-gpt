"""Incremental ingestion component for PrivateGPT.

This module bridges the standalone IncrementalUpdater with PrivateGPT's
ingestion pipeline.  It implements ``BaseIngestComponentWithIndex`` so that
the rest of the application (IngestService, API, UI) can use incremental
updates transparently - no call-site changes required.

When ``embedding.ingest_mode`` is set to ``"incremental"`` in the settings
YAML, ``get_ingestion_component()`` returns an instance of this class.  All
ingestion requests are then routed through the IncrementalUpdater, which:

* Chunks with SemanticChunker (paragraph-boundary splitting)
* Detects changes via DiffDetector (hash + Ratcliff/Obershelp)
* Re-embeds only ADDED and MODIFIED chunks
* Updates the VectorStoreIndex via insert_nodes / delete_nodes
* Persists chunk hashes in ChunkHashStore for future diffs

References (from thesis):
- thesis: Methodology -- proof-of-concept design and implementation
- thesis: Problem statement -- inefficiency of full re-ingestion
"""

import logging
from pathlib import Path
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.schema import Document, TransformComponent
from llama_index.core.storage import StorageContext

from private_gpt.components.ingest.incremental.chunk_hasher import SemanticChunker
from private_gpt.components.ingest.incremental.diff_detector import DiffDetector
from private_gpt.components.ingest.incremental.incremental_updater import (
    IncrementalUpdater,
)
from private_gpt.components.ingest.ingest_component import (
    BaseIngestComponentWithIndex,
)
from private_gpt.paths import local_data_path

logger = logging.getLogger(__name__)


class IncrementalIngestComponent(BaseIngestComponentWithIndex):
    """Drop-in ingest component that uses incremental chunk-level updates.

    This class wraps :class:`IncrementalUpdater` behind the standard
    ``BaseIngestComponentWithIndex`` interface so the rest of PrivateGPT
    does not need to know about the incremental logic.

    Parameters
    ----------
    storage_context : StorageContext
        Standard LlamaIndex storage context.
    embed_model : EmbedType
        The embedding model (used for computing embeddings of new/modified chunks).
    transformations : list[TransformComponent]
        LlamaIndex transformation pipeline (must include the embedding model).
    min_chunk_size : int
        Minimum characters per semantic chunk (default from settings).
    max_chunk_size : int
        Maximum characters per semantic chunk (default from settings).
    similarity_threshold : float
        Minimum Ratcliff/Obershelp ratio to pair modified chunks.
    """

    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        *args: Any,
        min_chunk_size: int = 100,
        max_chunk_size: int = 3000,
        similarity_threshold: float = 0.4,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, embed_model, transformations, *args, **kwargs)

        chunker = SemanticChunker(
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )
        diff_detector = DiffDetector(
            similarity_threshold=similarity_threshold,
        )

        # The IncrementalUpdater does its own semantic chunking, so we must
        # NOT pass node-parser transforms (like SentenceWindowNodeParser)
        # through to run_transformations() — that would re-split our chunks.
        # We only keep embedding transforms.
        embed_only_transforms: list[TransformComponent] = [
            t for t in self.transformations if isinstance(t, BaseEmbedding)
        ]

        # Pass the index already initialised by BaseIngestComponentWithIndex
        # to avoid loading the same VectorStoreIndex a second time.
        self._updater = IncrementalUpdater(
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            transformations=embed_only_transforms,
            persist_dir=local_data_path,
            chunker=chunker,
            diff_detector=diff_detector,
            index=self._index,
        )

        logger.info(
            "IncrementalIngestComponent initialised "
            "(min_chunk=%d, max_chunk=%d, sim_thresh=%.2f, "
            "embed_transforms=%d of %d total)",
            min_chunk_size,
            max_chunk_size,
            similarity_threshold,
            len(embed_only_transforms),
            len(self.transformations),
        )

    # ── BaseIngestComponent interface ────────────────────────────────

    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        """Ingest a single file incrementally.

        If the file was ingested before, only changed chunks are
        re-embedded.  Returns the LlamaIndex Documents that were
        processed (for compatibility with the IngestService response).
        """
        logger.info("Incremental ingest for file_name=%s", file_name)
        with self._index_thread_lock:
            stats, documents = self._updater.ingest_file(file_name, file_data)
            logger.info(
                "Incremental ingest complete: %s",
                stats.summary().replace("\n", " | "),
            )

        # Return the Documents already parsed by the updater so the API
        # layer can build IngestedDoc objects.  No need to re-parse the file.
        return documents

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        """Ingest multiple files incrementally.

        Each file is processed sequentially through the incremental
        pipeline.  This is intentional - the incremental approach saves
        time by skipping unchanged chunks rather than by parallelising
        file reads.
        """
        all_documents: list[Document] = []
        for file_name, file_data in files:
            try:
                documents = self.ingest(file_name, file_data)
                all_documents.extend(documents)
            except Exception:
                logger.exception("Failed incremental ingest for file=%s", file_name)
        return all_documents

    def delete(self, doc_id: str) -> None:
        """Delete a document by its doc_id.

        Tries the incremental hash-store first (by doc_id, then by
        file_name) which knows all chunk node_ids.  Falls back to the
        standard LlamaIndex ref_doc deletion for files that were ingested
        before incremental mode was enabled.
        """
        with self._index_thread_lock:
            # Try direct doc_id lookup in the hash store.
            record = self._updater.hash_store.get_document(doc_id)
            if record is not None:
                self._updater.delete_file(record.file_name)
                return

            # The UI / API may pass a doc_id that is actually a file_name,
            # e.g. when the record was created with file_name as doc_id.
            record = self._updater.hash_store.get_document_by_filename(doc_id)
            if record is not None:
                self._updater.delete_file(record.file_name)
                return

            # Fallback: the file was ingested the old way (pre-incremental).
            logger.info(
                "doc_id=%s not in hash store — falling back to ref_doc delete",
                doc_id,
            )
            try:
                self._index.delete_ref_doc(doc_id, delete_from_docstore=True)
                self._save_index()
            except Exception as e:
                # The vector store and docstore are out of sync (nodes missing
                # from the vector store, or already deleted).  Always fall back
                # to removing the docstore entry so the file no longer appears
                # in the ingested-files list.
                logger.warning(
                    "delete_ref_doc failed for doc_id=%s (%s) — "
                    "removing from docstore only",
                    doc_id,
                    e,
                )
                try:
                    # Bypass VectorStoreIndex._delete_from_index_struct which
                    # throws KeyError when nodes were inserted by a different
                    # VectorStoreIndex instance.  Go straight to the backends.
                    self._index.storage_context.vector_store.delete(doc_id)
                    self._index.docstore.delete_ref_doc(doc_id, raise_error=False)
                    self._save_index()
                except Exception:
                    pass  # Already gone — nothing to clean up
