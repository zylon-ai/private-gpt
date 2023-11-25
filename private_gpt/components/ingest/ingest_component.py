import abc
import itertools
import logging
import multiprocessing
import multiprocessing.pool
import os
import threading
from pathlib import Path
from typing import Any

from llama_index import (
    Document,
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.data_structs import IndexDict
from llama_index.indices.base import BaseIndex
from llama_index.ingestion import run_transformations

from private_gpt.components.ingest.ingest_helper import IngestionHelper
from private_gpt.paths import local_data_path

logger = logging.getLogger(__name__)


class BaseIngestComponent(abc.ABC):
    def __init__(
        self,
        storage_context: StorageContext,
        service_context: ServiceContext,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        logger.debug("Initializing base ingest component type=%s", type(self).__name__)
        self.storage_context = storage_context
        self.service_context = service_context

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
        service_context: ServiceContext,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, service_context, *args, **kwargs)

        self.show_progress = True
        self._index_thread_lock = (
            threading.RLock()
        )  # Thread lock! Not Multiprocessing lock
        self._index = self._initialize_index()

    def _initialize_index(self) -> BaseIndex[IndexDict]:
        """Initialize the index from the storage context."""
        try:
            # Load the index with store_nodes_override=True to be able to delete them
            index = load_index_from_storage(
                storage_context=self.storage_context,
                service_context=self.service_context,
                store_nodes_override=True,  # Force store nodes in index and document stores
                show_progress=self.show_progress,
            )
        except ValueError:
            # There are no index in the storage context, creating a new one
            logger.info("Creating a new vector store index")
            index = VectorStoreIndex.from_documents(
                [],
                storage_context=self.storage_context,
                service_context=self.service_context,
                store_nodes_override=True,  # Force store nodes in index and document stores
                show_progress=self.show_progress,
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
        service_context: ServiceContext,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, service_context, *args, **kwargs)

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


class MultiWorkerIngestComponent(BaseIngestComponentWithIndex):
    """Parallelize the file reading and parsing on multiple CPU core.

    This also makes the embeddings to be computed in batches (on GPU or CPU).
    """

    BULK_INGEST_WORKER_NUM = max((os.cpu_count() or 1) - 1, 1)

    def __init__(
        self,
        storage_context: StorageContext,
        service_context: ServiceContext,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, service_context, *args, **kwargs)
        # Make an efficient use of the CPU and GPU, the embedding
        # must be in the transformations
        assert (
            len(self.service_context.transformations) >= 2
        ), "Embeddings must be in the transformations"

    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        logger.info("Ingesting file_name=%s", file_name)
        documents = IngestionHelper.transform_file_into_documents(file_name, file_data)
        logger.info(
            "Transformed file=%s into count=%s documents", file_name, len(documents)
        )
        logger.debug("Saving the documents in the index and doc store")
        return self._save_docs(documents)

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        with multiprocessing.Pool(processes=self.BULK_INGEST_WORKER_NUM) as pool:
            documents = list(
                itertools.chain.from_iterable(
                    pool.starmap(IngestionHelper.transform_file_into_documents, files)
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
            self.service_context.transformations,
            show_progress=self.show_progress,
        )
        # Locking the index to avoid concurrent writes
        with self._index_thread_lock:
            logger.debug("Inserting count=%s nodes in the index", len(nodes))
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

    FIXME: this is not working as well as planned because of the usage of
     the multiprocessing worker pool.
    """

    BULK_INGEST_WORKER_NUM = max((os.cpu_count() or 1) - 1, 1)

    def __init__(
        self,
        storage_context: StorageContext,
        service_context: ServiceContext,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, service_context, *args, **kwargs)
        # Make an efficient use of the CPU and GPU, the embedding
        # must be in the transformations
        assert (
            len(self.service_context.transformations) >= 2
        ), "Embeddings must be in the transformations"
        # We are doing our own multiprocessing
        # To do not collide with the multiprocessing of huggingface, we disable it
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def ingest(self, file_name: str, file_data: Path) -> list[Document]:
        logger.info("Ingesting file_name=%s", file_name)
        # FIXME there are some cases where the process is not finished
        #  causing deadlocks. More information using trace:
        #  time PGPT_PROFILES=ingest-local python -m trace --trace \
        #    ./scripts/ingest_folder.py ... &> ingestion.traces
        with multiprocessing.Pool(processes=1) as pool:
            # Running in a single (1) process to release the current
            # thread, and take a dedicated CPU core for computation
            a_documents = pool.apply_async(
                IngestionHelper.transform_file_into_documents, (file_name, file_data)
            )
            while True:
                # FIXME ugly hack to highlight the deadlock in traces
                try:
                    documents = list(a_documents.get(timeout=2))
                except multiprocessing.TimeoutError:
                    continue
                break
            pool.close()
            pool.terminate()
        logger.info(
            "Transformed file=%s into count=%s documents", file_name, len(documents)
        )
        logger.debug("Saving the documents in the index and doc store")
        return self._save_docs(documents)

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        # Lightweight threads, used for parallelize the
        # underlying IO calls made in the ingestion
        with multiprocessing.pool.ThreadPool(
            processes=self.BULK_INGEST_WORKER_NUM
        ) as pool:
            documents = list(
                itertools.chain.from_iterable(pool.starmap(self.ingest, files))
            )
        return documents

    def _save_docs(self, documents: list[Document]) -> list[Document]:
        logger.debug("Transforming count=%s documents into nodes", len(documents))
        nodes = run_transformations(
            documents,  # type: ignore[arg-type]
            self.service_context.transformations,
            show_progress=self.show_progress,
        )
        # Locking the index to avoid concurrent writes
        with self._index_thread_lock:
            logger.debug("Inserting count=%s nodes in the index", len(nodes))
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
