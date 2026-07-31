"""Filesystem event watcher for the ZGPT file callback system (T5.1).

Observes mounted namespace directories for changes, deduplicates noise
(rapid repeated writes, placeholders, multipart temp files), and emits
one typed FileEvent per logical save.

Deduplication strategy:
- Events within DEBOUNCE_SECONDS of each other for the same path are
  collapsed into a single event.
- Paths whose filenames contain patterns suggesting temp/partial files
  (e.g. `~`, `.tmp`, `.part`, `.crdownload`, `.download`) are skipped.
- Empty files (zero bytes) are skipped unless the event is a delete.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from watchdog.events import (
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from private_gpt.components.filesystems.callbacks.models import FileEvent

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from watchdog.events import FileSystemEvent

logger = logging.getLogger(__name__)

# Seconds within which duplicate events for the same path are collapsed.
DEBOUNCE_SECONDS = 0.5

# Filename patterns indicating temp / partial files — skip these.
_TEMP_PATTERNS = re.compile(
    r"(^~|\.tmp$|\.part$|\.crdownload$|\.download$|^\.~|~$)",
    re.IGNORECASE,
)


def _is_temp_file(path: str) -> bool:
    name = os.path.basename(path)
    return bool(_TEMP_PATTERNS.search(name))


class _DebouncedHandler(FileSystemEventHandler):
    """Watchdog handler that deduplicates events and fires an async callback."""

    def __init__(
        self,
        namespace: str,
        scope: str,
        root: Path,
        correlation: dict[str, Any],
        emit: Callable[[FileEvent], Coroutine[Any, Any, None]],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__()
        self._namespace = namespace
        self._scope = scope
        self._root = root
        self._correlation = correlation
        self._emit = emit
        self._loop = loop
        # path → (timestamp, event_type) for debouncing
        self._pending: dict[str, tuple[float, str]] = defaultdict(lambda: (0.0, ""))

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle(event.src_path, "file.created")

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle(event.src_path, "file.updated")

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle(event.src_path, "file.deleted")

    def _handle(self, abs_path: str, event_type: str) -> None:
        if _is_temp_file(abs_path):
            logger.debug("Ignoring temp file event: %s", abs_path)
            return

        # Skip empty files for created/updated — zero-byte placeholders
        if event_type != "file.deleted":
            try:
                if os.path.getsize(abs_path) == 0:
                    logger.debug("Skipping zero-byte file event: %s", abs_path)
                    return
            except OSError:
                return  # File already gone

        now = time.monotonic()
        prev_ts, prev_type = self._pending[abs_path]
        if now - prev_ts < DEBOUNCE_SECONDS and prev_type == event_type:
            # Duplicate — update timestamp but don't re-fire
            self._pending[abs_path] = (now, event_type)
            return

        self._pending[abs_path] = (now, event_type)

        try:
            rel = str(Path(abs_path).relative_to(self._root))
        except ValueError:
            rel = abs_path

        ev = FileEvent(
            type=event_type,  # type: ignore[arg-type]
            path=rel,
            namespace=self._namespace,
            scope=self._scope,
            correlation=self._correlation,
        )
        # Schedule the coroutine on the event loop (watchdog runs on a thread)
        asyncio.run_coroutine_threadsafe(self._emit(ev), self._loop)


class FileWatchSession:
    """An active filesystem watch session for one namespace/scope pair.

    Call start() to begin observing, stop() to release the observer.
    """

    def __init__(
        self,
        namespace: str,
        scope: str,
        watch_path: Path,
        correlation: dict[str, Any],
        emit: Callable[[FileEvent], Coroutine[Any, Any, None]],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._watch_path = watch_path
        self._handler = _DebouncedHandler(
            namespace=namespace,
            scope=scope,
            root=watch_path,
            correlation=correlation,
            emit=emit,
            loop=loop,
        )
        self._observer = Observer()

    def start(self) -> None:
        if not self._watch_path.exists():
            logger.warning("Watch path does not exist, skipping: %s", self._watch_path)
            return
        self._observer.schedule(self._handler, str(self._watch_path), recursive=True)
        self._observer.start()
        logger.debug("Started watching: %s", self._watch_path)

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join(timeout=2)
        logger.debug("Stopped watching: %s", self._watch_path)
