import argparse
import sys
from pathlib import Path

from loguru import logger

from private_gpt.di import root_injector
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.ingest.ingest_watcher import IngestWatcher

ingest_service = root_injector.get(IngestService)

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

# Set up loguru logging
# Remove pre-configured logging handler
logger.remove(0)
# For console colorized output without line and function info:
logger.add(
    sys.stdout,
    level="INFO",
    colorize=True,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    ),
)
# For file output, using a clear and timestamped format:
if args.log_file:
    logger.add(
        args.log_file,
        rotation="10 MB",
        level="INFO",
        format="[{time:YYYY-MM-DD HH:mm:ss}] [{level}] {message}",
    )

total_documents = 0
current_document_count = 0


def count_documents(folder_path: Path) -> None:
    global total_documents
    for file_path in folder_path.iterdir():
        if file_path.is_file():
            total_documents += 1
        elif file_path.is_dir():
            count_documents(file_path)


def _recursive_ingest_folder(folder_path: Path) -> None:
    global current_document_count, total_documents
    for file_path in folder_path.iterdir():
        if file_path.is_file():
            current_document_count += 1
            progress_msg = f"Document {current_document_count} of {total_documents} ({(current_document_count / total_documents) * 100:.2f}%)"
            logger.info(progress_msg)
            _do_ingest(file_path)
        elif file_path.is_dir():
            _recursive_ingest_folder(file_path)


def _do_ingest(changed_path: Path) -> None:
    try:
        if changed_path.exists():
            logger.info(f"Started ingesting {changed_path}")
            ingest_service.ingest(changed_path.name, changed_path)
            logger.info(f"Completed ingesting {changed_path}")
    except Exception as e:
        logger.error(f"Failed to ingest document: {changed_path}. Error: {e}")


path = Path(args.folder)
if not path.exists():
    raise ValueError(f"Path {args.folder} does not exist")

# Count total documents before ingestion
count_documents(path)

_recursive_ingest_folder(path)
if args.watch:
    logger.info(f"Watching {args.folder} for changes, press Ctrl+C to stop...")
    watcher = IngestWatcher(args.folder, _do_ingest)
    watcher.start()
