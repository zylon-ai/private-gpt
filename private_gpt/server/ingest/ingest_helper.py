import itertools
import logging
import multiprocessing
import os
import queue
from multiprocessing import Process
from pathlib import Path
from threading import Thread
from typing import Callable, Any

from llama_index import Document

from llama_index.readers import JSONReader, StringIterableReader
from llama_index.readers.file.base import DEFAULT_FILE_READER_CLS

# Patching the default file reader to support other file types
FILE_READER_CLS = DEFAULT_FILE_READER_CLS.copy()
FILE_READER_CLS.update(
    {
        ".json": JSONReader,
    }
)

logger = logging.getLogger(__name__)


class IngestionHelper:
    """
    Helper class to transform a file into a list of documents.

    This class should be used to transform a file into a list of documents.
    These methods are thread-safe (and multiprocessing-safe).
    """
    @staticmethod
    def transform_file_into_documents(
        file_name: str, file_data: Path
    ) -> list[Document]:
        documents = IngestionHelper._load_file_to_documents(file_name, file_data)
        for document in documents:
            document.metadata["file_name"] = file_name
        IngestionHelper._exclude_metadata(documents)
        return documents

    @staticmethod
    def _load_file_to_documents(file_name: str, file_data: Path) -> list[Document]:
        logger.debug("Transforming file_name=%s into documents", file_name)
        extension = Path(file_name).suffix
        reader_cls = FILE_READER_CLS.get(extension)
        if reader_cls is None:
            logger.debug(
                "No reader found for extension=%s, using default string reader",
                extension,
            )
            # Read as a plain text
            string_reader = StringIterableReader()
            return string_reader.load_data([file_data.read_text()])

        logger.debug("Specific reader found for extension=%s", extension)
        return reader_cls().load_data(file_data)

    @staticmethod
    def _exclude_metadata(documents: list[Document]) -> None:
        for document in documents:
            document.metadata["doc_id"] = document.doc_id
            # We don't want the Embeddings search to receive this metadata
            document.excluded_embed_metadata_keys = ["doc_id"]
            # We don't want the LLM to receive these metadata in the context
            document.excluded_llm_metadata_keys = ["file_name", "doc_id", "page_label"]


def _file_to_documents_worker_main(job_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue) -> None:
    """Worker main for the file to documents worker.

    This worker reads from the job queue, and writes to the result queue.
    Follows the https://docs.python.org/3/library/multiprocessing.html#programming-guidelines
    """
    result_queue.cancel_join_thread()  # Don't wait for the result queue to be empty to exit
    while True:
        job = job_queue.get()
        if job is None:
            break
        file_name, file_path = job
        documents = IngestionHelper.transform_file_into_documents(file_name, file_path)
        logger.info("Transformed file_name=%s into count=%s documents", file_name, len(documents))
        result_queue.put(documents, block=True)  # Wait if the queue is full
    logger.debug("No more work, file to documents worker exiting")


def _documents_to_db_worker_main(job_queue: multiprocessing.Queue, to_db_func: Callable[[list[Document]], Any]) -> None:
    """Worker main for the documents to db worker.

    This is not safe to use multiple process (multiprocessing) workers, as
    to_db_func might not be multiprocessing-safe.

    This worker reads from the job queue, and calls the to_db_func.
    """
    while True:
        try:
            job = job_queue.get(block=True, timeout=2)
        except queue.Empty:
            logger.debug("Waiting additional documents to push to the DB")
            continue
        except ValueError:
            # Raised when the queue is closed
            break
        if job is None:
            break
        documents = job
        to_db_func(documents)
    logger.debug("No more work, documents to db worker exiting")

# TODO handle errors in a process

# TODO OR MAKE IT USE pool.astartmap() -> self._save_doc


class SimpleBulkIngestPipeline:

    def __init__(self):
        # Not more than 4 workers, not less than 1
        self.worker_count = None  # max(min(os.cpu_count() - 1, 4), 1)

    @staticmethod
    def _wrap_transform_file_into_documents(arg: tuple[str, Path]) -> list[Document]:
        return IngestionHelper.transform_file_into_documents(arg[0], arg[1])

    def bulk_ingest(
            self,
            files_to_process: list[tuple[str, Path]],
            to_db_func: Callable[[list[Document]], Any]) -> None:

        with multiprocessing.Pool(processes=self.worker_count) as pool:
            file_to_documents_converter = pool.map(
                SimpleBulkIngestPipeline._wrap_transform_file_into_documents,
                files_to_process,
                chunksize=1)
            to_db_func(list(itertools.chain.from_iterable(file_to_documents_converter)))
            # for documents in file_to_documents_converter:
            #     to_db_func(documents)


class BulkIngestPipeline:
    """Pipeline for bulk ingest.

    Composed of two queues, and one multiprocess read-worker pool.

    The first queue contains the file Path to ingest, which is read by
    the read-worker pool.
    The second queue contains the data read and converted as documents
    from the file, which is written by the read-worker pool. This queue is
    limited in size, to reduce memory usage.

    Lastly, there is a single worker thread pool that does the insertion
    into the database. This is done in a single thread to reduce the
    number of connections to the database (as DB connections might not be thread-safe).
    """

    def __init__(self, documents_queue_max_size: int = 4) -> None:

        # Number of workers
        self.file_to_documents_workers_count = max(min(os.cpu_count() - 2, documents_queue_max_size), 1)

        # Multiprocess pools
        self.file_queue: multiprocessing.JoinableQueue[tuple[str, Path]] = multiprocessing.JoinableQueue(maxsize=0)
        self.documents_queue: multiprocessing.JoinableQueue[list[Document]] = multiprocessing.JoinableQueue(maxsize=documents_queue_max_size)

        self.file_to_documents_workers: list[Process] = []
        self.documents_to_db_workers: list[Thread] = []

    def bulk_ingest(
            self,
            files_to_process: list[tuple[str, Path]],
            to_db_func: Callable[[list[Document]], Any]) -> None:
        """Bulk ingest the files at the given paths.

        Args:
            files_to_process (list[str]): List of file paths to ingest.
            to_db_func (Callable[[Document], None]): Function to call to insert the document into the database.
                It must be thread safe, but is not required to be multiprocessing-safe.
        """
        self.file_to_documents_workers = [
            Process(
                # File To Documents worker
                name=f"ftd_{i}",
                target=IngestionHelper.transform_file_into_documents,
                kwargs={"job_queue": self.file_queue, "result_queue": self.documents_queue},
            ) for i in range(self.file_to_documents_workers_count)
        ]

        self.documents_to_db_workers = [Thread(
            # Documents To DB worker
            name=f"dtb_{i}",
            target=to_db_func,
            args=(),
        ) for i in range(1)]

        self._initialize_file_queue(files_to_process)
        for ftd_worker in self.file_to_documents_workers:
            ftd_worker.start()

        for dtb_worker in self.documents_to_db_workers:
            dtb_worker.start()

        self.file_queue.join()
        self.file_queue.close()

        for ftd_worker in self.file_to_documents_workers:
            ftd_worker.join()

        self.documents_queue.join()
        # We class the queue to prevent the documents to db worker from waiting
        self.documents_queue.close()

        for dtb_worker in self.documents_to_db_workers:
            dtb_worker.join()

    def _initialize_file_queue(self, files_to_process: list[tuple[str, Path]]) -> None:
        for filename, file_path in files_to_process:
            self.file_queue.put((filename, file_path))
