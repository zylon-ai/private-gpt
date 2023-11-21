import argparse
import logging
from pathlib import Path

from private_gpt.di import global_injector
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.ingest.ingest_watcher import IngestWatcher

logger = logging.getLogger(__name__)

ingest_service = global_injector.get(IngestService)

parser = argparse.ArgumentParser(prog="ingest_folder.py")
parser.add_argument("folder", help="Folder to ingest")
parser.add_argument(
    "--watch",
    help="Watch for changes",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument(
    "--log-file",
    help="Optional path to a log file. If provided, logs will be written to this file.",
    type=str,
    default=None,
)
args = parser.parse_args()

# Set up logging to a file if a path is provided
if args.log_file:
    file_handler = logging.FileHandler(args.log_file, mode="a")
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)


class LocalIngestWorker:
    def __init__(self, ingest_service: IngestService) -> None:
        self.ingest_service = ingest_service

        self.total_documents = 0
        self.current_document_count = 0

    def ingest(self, file_path):
        ingest_service.ingest(file_path.name, file_path)

    def count_documents(self, folder_path: Path) -> None:
        for file_path in folder_path.iterdir():
            if file_path.is_file():
                self.total_documents += 1
            elif file_path.is_dir():
                self.count_documents(file_path)

    def ingest_folder(self, folder_path: Path) -> None:
        # Count total documents before ingestion
        self.count_documents(folder_path)
        self._recursive_ingest_folder(folder_path)

    # TODO refactor to parse the file tree, and transform it into a list of files to ingest
    #      then, use the ingest_service.bulk_ingest method
    def _recursive_ingest_folder(self, folder_path: Path) -> None:
        for file_path in folder_path.iterdir():
            if file_path.is_file():
                self.current_document_count += 1
                pct = f"{(self.current_document_count / self.total_documents) * 100:.2f}"
                progress_msg = f"Document {self.current_document_count} of {self.total_documents} ({pct}%)"
                logger.info(progress_msg)
                self._do_ingest(file_path)
            elif file_path.is_dir():
                self._recursive_ingest_folder(file_path)

    def ingest_on_watch(self, changed_path: Path) -> None:
        logger.info("Detected change in at path=%s, ingesting", changed_path)
        self._do_ingest(changed_path)

    def _do_ingest(self, changed_path: Path) -> None:
        try:
            if changed_path.exists():
                logger.info(f"Started ingesting {changed_path}")
                self.ingest_service.ingest(changed_path.name, changed_path)
                logger.info(f"Completed ingesting {changed_path}")
        except Exception:
            logger.exception(
                f"Failed to ingest document: {changed_path}, find the exception attached"
            )


path = Path(args.folder)
if not path.exists():
    raise ValueError(f"Path {args.folder} does not exist")

worker = LocalIngestWorker(ingest_service)
worker.ingest_folder(path)

if args.watch:
    logger.info(f"Watching {args.folder} for changes, press Ctrl+C to stop...")
    watcher = IngestWatcher(args.folder, worker.ingest_on_watch)
    watcher.start()
