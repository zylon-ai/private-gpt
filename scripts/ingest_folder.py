import argparse
import sys
from pathlib import Path

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
args = parser.parse_args()


def _recursive_ingest_folder(folder_path: Path) -> None:
    for file_path in folder_path.iterdir():
        if file_path.is_file():
            _do_ingest(file_path)
        elif file_path.is_dir():
            _recursive_ingest_folder(file_path)


def _do_ingest(changed_path: Path) -> None:
    if changed_path.exists():
        print(f"\nIngesting {changed_path}")
        ingest_service.ingest(changed_path.name, changed_path)


path = Path(args.folder)
if not path.exists():
    raise ValueError(f"Path {args.folder} does not exist")

_recursive_ingest_folder(path)
if args.watch:
    print(f"Watching {args.folder} for changes, press Ctrl+C to stop...")
    watcher = IngestWatcher(args.folder, _do_ingest)
    watcher.start()
