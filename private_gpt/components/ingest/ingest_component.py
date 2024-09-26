import abc
import itertools
import logging
import multiprocessing
import multiprocessing.pool
import os
import threading
from pathlib import Path
from queue import Queue
from typing import Any

from llama_index.core.data_structs import IndexDict
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.indices import VectorStoreIndex, load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from llama_index.core.ingestion import run_transformations
from llama_index.core.schema import BaseNode, Document, TransformComponent
from llama_index.core.storage import StorageContext

from private_gpt.components.ingest.ingest_helper import IngestionHelper
from private_gpt.paths import local_data_path
from private_gpt.settings.settings import Settings
from private_gpt.utils.eta import eta

logger = logging.getLogger(__name__)


class BaseIngestComponent(abc.ABC):
    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        logger.debug("Initializing base ingest component type=%s", type(self).__name__)
        self.storage_context = storage_context
        self.embed_model = embed_model
        self.transformations = transformations

    @abc.abstractmethod
    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        pass

    @abc.abstractmethod
    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        pass

    @abc.abstractmethod
    def delete(self, doc_id: str) -> None:
        pass


class BaseIngestComponentWithIndex(BaseIngestComponent, abc.ABC):
    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, embed_model, transformations, *args, **kwargs)

        self.show_progress = True
        self._index_thread_lock = (
            threading.Lock()
        )  # Thread lock! Not Multiprocessing lock
        self._index = self._initialize_index()

    def _initialize_index(self) -> BaseIndex[IndexDict]:
        """Initialize the index from the storage context."""
        try:
            # Load the index with store_nodes_override=True to be able to delete them
            index = load_index_from_storage(
                storage_context=self.storage_context,
                store_nodes_override=True,  # Force store nodes in index and document stores
                show_progress=self.show_progress,
                embed_model=self.embed_model,
                transformations=self.transformations,
            )
        except ValueError:
            # There are no index in the storage context, creating a new one
            logger.info("Creating a new vector store index")
            index = VectorStoreIndex.from_documents(
                [],
                storage_context=self.storage_context,
                store_nodes_override=True,  # Force store nodes in index and document stores
                show_progress=self.show_progress,
                embed_model=self.embed_model,
                transformations=self.transformations,
            )
            index.storage_context.persist(persist_dir=local_data_path)
        return index

    def _save_index(self) -> None:
        self._index.storage_context.persist(persist_dir=local_data_path)

    def delete(self, doc_id: str) -> None:
        with self._index_thread_lock:
            # Delete the document from the index
            self._index.delete_ref_doc(doc_id, delete_from_docstore=True)

            # Save the index
            self._save_index()


class SimpleIngestComponent(BaseIngestComponentWithIndex):
    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, embed_model, transformations, *args, **kwargs)

    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        logger.info("Ingesting file_name=%s", file_name)
        documents = IngestionHelper.transform_file_into_documents(file_name, file_data)
        logger.info(
            "Transformed file=%s into count=%s documents", file_name, len(documents)
        )
        logger.debug("Saving the documents in the index and doc store")
        return self._save_docs(documents)

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        saved_documents = []
        for file_name, file_data in files:
            documents = IngestionHelper.transform_file_into_documents(
                file_name, file_data
            )
            saved_documents.extend(self._save_docs(documents))
        return saved_documents

    def _save_docs(self, documents: list[Document]) -> list[Document]:
        logger.debug("Transforming count=%s documents into nodes", len(documents))
        with self._index_thread_lock:
            for document in documents:
                self._index.insert(document, show_progress=True)
            logger.debug("Persisting the index and nodes")
            # persist the index and nodes
            self._save_index()
            logger.debug("Persisted the index and nodes")
        return documents


class BatchIngestComponent(BaseIngestComponentWithIndex):
    """Parallelize the file reading and parsing on multiple CPU core.

    This also makes the embeddings to be computed in batches (on GPU or CPU).
    """

    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        count_workers: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, embed_model, transformations, *args, **kwargs)
        # Make an efficient use of the CPU and GPU, the embedding
        # must be in the transformations
        assert (
            len(self.transformations) >= 2
        ), "Embeddings must be in the transformations"
        assert count_workers > 0, "count_workers must be > 0"
        self.count_workers = count_workers

        self._file_to_documents_work_pool = multiprocessing.Pool(
            processes=self.count_workers
        )

    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        logger.info("Ingesting file_name=%s", file_name)
        documents = IngestionHelper.transform_file_into_documents(file_name, file_data)
        logger.info(
            "Transformed file=%s into count=%s documents", file_name, len(documents)
        )
        logger.debug("Saving the documents in the index and doc store")
        return self._save_docs(documents)

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        documents = list(
            itertools.chain.from_iterable(
                self._file_to_documents_work_pool.starmap(
                    IngestionHelper.transform_file_into_documents, files
                )
            )
        )
        logger.info(
            "Transformed count=%s files into count=%s documents",
            len(files),
            len(documents),
        )
        return self._save_docs(documents)

    def _save_docs(self, documents: list[Document]) -> list[Document]:
        logger.debug("Transforming count=%s documents into nodes", len(documents))
        nodes = run_transformations(
            documents,  # type: ignore[arg-type]
            self.transformations,
            show_progress=self.show_progress,
        )
        # Locking the index to avoid concurrent writes
        with self._index_thread_lock:
            logger.info("Inserting count=%s nodes in the index", len(nodes))
            self._index.insert_nodes(nodes, show_progress=True)
            for document in documents:
                self._index.docstore.set_document_hash(
                    document.get_doc_id(), document.hash
                )
            logger.debug("Persisting the index and nodes")
            # persist the index and nodes
            self._save_index()
            logger.debug("Persisted the index and nodes")
        return documents


class ParallelizedIngestComponent(BaseIngestComponentWithIndex):
    """Parallelize the file ingestion (file reading, embeddings, and index insertion).

    This use the CPU and GPU in parallel (both running at the same time), and
    reduce the memory pressure by not loading all the files in memory at the same time.
    """

    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        count_workers: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, embed_model, transformations, *args, **kwargs)
        # To make an efficient use of the CPU and GPU, the embeddings
        # must be in the transformations (to be computed in batches)
        assert (
            len(self.transformations) >= 2
        ), "Embeddings must be in the transformations"
        assert count_workers > 0, "count_workers must be > 0"
        self.count_workers = count_workers
        # We are doing our own multiprocessing
        # To do not collide with the multiprocessing of huggingface, we disable it
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self._ingest_work_pool = multiprocessing.pool.ThreadPool(
            processes=self.count_workers
        )

        self._file_to_documents_work_pool = multiprocessing.Pool(
            processes=self.count_workers
        )

    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        logger.info("Ingesting file_name=%s", file_name)
        # Running in a single (1) process to release the current
        # thread, and take a dedicated CPU core for computation
        documents = self._file_to_documents_work_pool.apply(
            IngestionHelper.transform_file_into_documents, (file_name, file_data)
        )
        logger.info(
            "Transformed file=%s into count=%s documents", file_name, len(documents)
        )
        logger.debug("Saving the documents in the index and doc store")
        return self._save_docs(documents)

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        # Lightweight threads, used for parallelize the
        # underlying IO calls made in the ingestion

        documents = list(
            itertools.chain.from_iterable(
                self._ingest_work_pool.starmap(self.ingest, files)
            )
        )
        return documents

    def _save_docs(self, documents: list[Document]) -> list[Document]:
        logger.debug("Transforming count=%s documents into nodes", len(documents))
        nodes = run_transformations(
            documents,  # type: ignore[arg-type]
            self.transformations,
            show_progress=self.show_progress,
        )
        # Locking the index to avoid concurrent writes
        with self._index_thread_lock:
            logger.info("Inserting count=%s nodes in the index", len(nodes))
            self._index.insert_nodes(nodes, show_progress=True)
            for document in documents:
                self._index.docstore.set_document_hash(
                    document.get_doc_id(), document.hash
                )
            logger.debug("Persisting the index and nodes")
            # persist the index and nodes
            self._save_index()
            logger.debug("Persisted the index and nodes")
        return documents

    def __del__(self) -> None:
        # We need to do the appropriate cleanup of the multiprocessing pools
        # when the object is deleted. Using root logger to avoid
        # the logger to be deleted before the pool
        logging.debug("Closing the ingest work pool")
        self._ingest_work_pool.close()
        self._ingest_work_pool.join()
        self._ingest_work_pool.terminate()
        logging.debug("Closing the file to documents work pool")
        self._file_to_documents_work_pool.close()
        self._file_to_documents_work_pool.join()
        self._file_to_documents_work_pool.terminate()


class PipelineIngestComponent(BaseIngestComponentWithIndex):
    """Pipeline ingestion - keeping the embedding worker pool as busy as possible.

    This class implements a threaded ingestion pipeline, which comprises two threads
    and two queues. The primary thread is responsible for reading and parsing files
    into documents. These documents are then placed into a queue, which is
    distributed to a pool of worker processes for embedding computation. After
    embedding, the documents are transferred to another queue where they are
    accumulated until a threshold is reached. Upon reaching this threshold, the
    accumulated documents are flushed to the document store, index, and vector
    store.

    Exception handling ensures robustness against erroneous files. However, in the
    pipelined design, one error can lead to the discarding of multiple files. Any
    discarded files will be reported.
    """

    NODE_FLUSH_COUNT = 5000  # Save the index every # nodes.

    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        count_workers: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, embed_model, transformations, *args, **kwargs)
        self.count_workers = count_workers
        assert (
            len(self.transformations) >= 2
        ), "Embeddings must be in the transformations"
        assert count_workers > 0, "count_workers must be > 0"
        self.count_workers = count_workers
        # We are doing our own multiprocessing
        # To do not collide with the multiprocessing of huggingface, we disable it
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # doc_q stores parsed files as Document chunks.
        # Using a shallow queue causes the filesystem parser to block
        # when it reaches capacity. This ensures it doesn't outpace the
        # computationally intensive embeddings phase, avoiding unnecessary
        # memory consumption.  The semaphore is used to bound the async worker
        # embedding computations to cause the doc Q to fill and block.
        self.doc_semaphore = multiprocessing.Semaphore(
            self.count_workers
        )  # limit the doc queue to # items.
        self.doc_q: Queue[tuple[str, str | None, list[Document] | None]] = Queue(20)
        # node_q stores documents parsed into nodes (embeddings).
        # Larger queue size so we don't block the embedding workers during a slow
        # index update.
        self.node_q: Queue[
            tuple[str, str | None, list[Document] | None, list[BaseNode] | None]
        ] = Queue(40)
        threading.Thread(target=self._doc_to_node, daemon=True).start()
        threading.Thread(target=self._write_nodes, daemon=True).start()

    def _doc_to_node(self) -> None:
        # Parse documents into nodes
        with multiprocessing.pool.ThreadPool(processes=self.count_workers) as pool:
            while True:
                try:
                    cmd, file_name, documents = self.doc_q.get(
                        block=True
                    )  # Documents for a file
                    if cmd == "process":
                        # Push CPU/GPU embedding work to the worker pool
                        # Acquire semaphore to control access to worker pool
                        self.doc_semaphore.acquire()
                        pool.apply_async(
                            self._doc_to_node_worker, (file_name, documents)
                        )
                    elif cmd == "quit":
                        break
                finally:
                    if cmd != "process":
                        self.doc_q.task_done()  # unblock Q joins

    def _doc_to_node_worker(self, file_name: str, documents: list[Document]) -> None:
        # CPU/GPU intensive work in its own process
        try:
            nodes = run_transformations(
                documents,  # type: ignore[arg-type]
                self.transformations,
                show_progress=self.show_progress,
            )
            self.node_q.put(("process", file_name, documents, list(nodes)))
        finally:
            self.doc_semaphore.release()
            self.doc_q.task_done()  # unblock Q joins

    def _save_docs(
        self, files: list[str], documents: list[Document], nodes: list[BaseNode]
    ) -> None:
        try:
            logger.info(
                f"Saving {len(files)} files ({len(documents)} documents / {len(nodes)} nodes)"
            )
            self._index.insert_nodes(nodes)
            for document in documents:
                self._index.docstore.set_document_hash(
                    document.get_doc_id(), document.hash
                )
            self._save_index()
        except Exception:
            # Tell the user so they can investigate these files
            logger.exception(f"Processing files {files}")
        finally:
            # Clearing work, even on exception, maintains a clean state.
            nodes.clear()
            documents.clear()
            files.clear()

    def _write_nodes(self) -> None:
        # Save nodes to index.  I/O intensive.
        node_stack: list[BaseNode] = []
        doc_stack: list[Document] = []
        file_stack: list[str] = []
        while True:
            try:
                cmd, file_name, documents, nodes = self.node_q.get(block=True)
                if cmd in ("flush", "quit"):
                    if file_stack:
                        self._save_docs(file_stack, doc_stack, node_stack)
                    if cmd == "quit":
                        break
                elif cmd == "process":
                    node_stack.extend(nodes)  # type: ignore[arg-type]
                    doc_stack.extend(documents)  # type: ignore[arg-type]
                    file_stack.append(file_name)  # type: ignore[arg-type]
                    # Constant saving is heavy on I/O - accumulate to a threshold
                    if len(node_stack) >= self.NODE_FLUSH_COUNT:
                        self._save_docs(file_stack, doc_stack, node_stack)
            finally:
                self.node_q.task_done()

    def _flush(self) -> None:
        self.doc_q.put(("flush", None, None))
        self.doc_q.join()
        self.node_q.put(("flush", None, None, None))
        self.node_q.join()

    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        documents = IngestionHelper.transform_file_into_documents(file_name, file_data)
        self.doc_q.put(("process", file_name, documents))
        self._flush()
        return documents

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        docs = []
        for file_name, file_data in eta(files):
            try:
                documents = IngestionHelper.transform_file_into_documents(
                    file_name, file_data
                )
                self.doc_q.put(("process", file_name, documents))
                docs.extend(documents)
            except Exception:
                logger.exception(f"Skipping {file_data.name}")
        self._flush()
        return docs


def get_ingestion_component(
    storage_context: StorageContext,
    embed_model: EmbedType,
    transformations: list[TransformComponent],
    settings: Settings,
) -> BaseIngestComponent:
    """Get the ingestion component for the given configuration."""
    ingest_mode = settings.embedding.ingest_mode
    if ingest_mode == "batch":
        return BatchIngestComponent(
            storage_context=storage_context,
            embed_model=embed_model,
            transformations=transformations,
            count_workers=settings.embedding.count_workers,
        )
    elif ingest_mode == "parallel":
        return ParallelizedIngestComponent(
            storage_context=storage_context,
            embed_model=embed_model,
            transformations=transformations,
            count_workers=settings.embedding.count_workers,
        )
    elif ingest_mode == "pipeline":
        return PipelineIngestComponent(
            storage_context=storage_context,
            embed_model=embed_model,
            transformations=transformations,
            count_workers=settings.embedding.count_workers,
        )
    else:
        return SimpleIngestComponent(
            storage_context=storage_context,
            embed_model=embed_model,
            transformations=transformations,
        )
