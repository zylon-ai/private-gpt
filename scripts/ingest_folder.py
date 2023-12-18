#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

from private_gpt.di import global_injector
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.ingest.ingest_watcher import IngestWatcher

logger = logging.getLogger(__name__)


class LocalIngestWorker:
    def __init__(self, ingest_service: IngestService) -> None:
        self.ingest_service = ingest_service

        self.total_documents = 0
        self.current_document_count = 0

        self._files_under_root_folder: list[Path] = list()

    def _find_all_files_in_folder(self, root_path: Path) -> None:
        """Search all files under the root folder recursively.
        Count them at the same time
        """
        for file_path in root_path.iterdir():
            if file_path.is_file():
                self.total_documents += 1
                self._files_under_root_folder.append(file_path)
            elif file_path.is_dir():
                self._find_all_files_in_folder(file_path)

    def ingest_folder(self, folder_path: Path) -> None:
        # Count total documents before ingestion
        self._find_all_files_in_folder(folder_path)
        self._ingest_all(self._files_under_root_folder)

    def _ingest_all(self, files_to_ingest: list[Path]) -> None:
        logger.info("Ingesting files=%s", [f.name for f in files_to_ingest])
        self.ingest_service.bulk_ingest([(str(p.name), p) for p in files_to_ingest])

    def ingest_on_watch(self, changed_path: Path) -> None:
        logger.info("Detected change in at path=%s, ingesting", changed_path)
        self._do_ingest_one(changed_path)

    def _do_ingest_one(self, changed_path: Path) -> None:
        try:
            if changed_path.exists():
                logger.info(f"Started ingesting file={changed_path}")
                self.ingest_service.ingest_file(changed_path.name, changed_path)
                logger.info(f"Completed ingesting file={changed_path}")
        except Exception:
            logger.exception(
                f"Failed to ingest document: {changed_path}, find the exception attached"
            )


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

if __name__ == "__main__":

    root_path = Path(args.folder)
    if not root_path.exists():
        raise ValueError(f"Path {args.folder} does not exist")

    ingest_service = global_injector.get(IngestService)
    worker = LocalIngestWorker(ingest_service)
    worker.ingest_folder(root_path)

    if args.watch:
        logger.info(f"Watching {args.folder} for changes, press Ctrl+C to stop...")
        watcher = IngestWatcher(args.folder, worker.ingest_on_watch)
        watcher.start()
