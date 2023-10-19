from collections.abc import Callable
from pathlib import Path
from typing import Any

from watchdog.events import (
    DirCreatedEvent,
    DirModifiedEvent,
    FileCreatedEvent,
    FileModifiedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer


class IngestWatcher:
    def __init__(
        self, watch_path: Path, on_file_changed: Callable[[Path], None]
    ) -> None:
        self.watch_path = watch_path
        self.on_file_changed = on_file_changed

        class Handler(FileSystemEventHandler):
            def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
                if isinstance(event, FileModifiedEvent):
                    on_file_changed(Path(event.src_path))

            def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
                if isinstance(event, FileCreatedEvent):
                    on_file_changed(Path(event.src_path))

        event_handler = Handler()
        observer: Any = Observer()
        self._observer = observer
        self._observer.schedule(event_handler, str(watch_path), recursive=True)

    def start(self) -> None:
        self._observer.start()
        while self._observer.is_alive():
            try:
                self._observer.join(1)
            except KeyboardInterrupt:
                break

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join()
