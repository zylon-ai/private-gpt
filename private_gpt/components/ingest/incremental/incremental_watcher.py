"""Per-file watcher with incremental update support.

Watches specific files (not a whole directory) for modifications and
routes changes through the incremental pipeline.

Unlike the previous directory-based approach, this watcher tracks individual
files that were explicitly ingested. Files can live anywhere on the filesystem.

Key features:
- Per-file registration: only files that have been ingested are watched
- Registrations survive stop/start: the file list is kept when the observer
  thread is stopped and restored when it is restarted
- Debouncing: rapid saves (e.g. editor autosave) produce only one event
- Thread-safe add/remove while the observer is running

References (from thesis):
- thesis: Methodology -- filewatcher module based on watchdog
- thesis: Problem statement -- --watch mode triggers full re-ingestion
"""

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from watchdog.events import (
    FileDeletedEvent,
    FileModifiedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

DEFAULT_DEBOUNCE_SECONDS = 2.0


class IncrementalIngestWatcher:
    """Per-file watcher that triggers incremental updates for registered files.

    Usage::

        watcher = IncrementalIngestWatcher(debounce_seconds=2.0)
        watcher.add_file_watch(Path("/docs/report.pdf"), on_modified=my_callback)
        watcher.start_background()   # start watching in a background thread
        ...
        watcher.stop()               # stops the thread; file list is preserved
        watcher.start_background()   # restarts and re-watches all registered files

    Parameters:
        debounce_seconds: Minimum time (s) between processing events for the
                          same file.  Prevents spurious re-ingestion from editors
                          that write a file multiple times on save.
    """

    def __init__(self, debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS) -> None:
        self.debounce_seconds = debounce_seconds

        # Persistent registration: file_path_str -> (on_modified, on_deleted|None)
        self._registered: dict[str, tuple[Callable[[Path], None], Callable[[Path], None] | None]] = {}
        self._reg_lock = threading.Lock()

        # Debounce tracking
        self._last_event_time: dict[str, float] = {}
        self._debounce_lock = threading.Lock()

        # The watchdog Observer (re-created on each start_background call)
        self._observer: Any = None
        # watchdog Watch objects: file_path_str -> Watch
        self._watch_handles: dict[str, Any] = {}

    # ── Registration ──────────────────────────────────────────────────

    def add_file_watch(
        self,
        file_path: Path,
        on_modified: Callable[[Path], None],
        on_deleted: Callable[[Path], None] | None = None,
    ) -> None:
        """Register a file for watching.

        If the observer is already running the file is scheduled immediately.
        Safe to call while the watcher is running.
        """
        resolved = file_path.resolve()
        key = str(resolved)

        # Pre-arm the debounce timer BEFORE checking if already registered.
        # This suppresses any filesystem event queued while the upload was
        # being copied — including re-uploads of an already-watched file where
        # the early-return below would otherwise skip the timer update and let
        # the copy event fire immediately.
        with self._debounce_lock:
            self._last_event_time[key] = time.time()

        with self._reg_lock:
            if key in self._registered:
                logger.debug("Already watching (debounce refreshed): %s", resolved)
                return
            self._registered[key] = (on_modified, on_deleted)

        if self._observer is not None and self._observer.is_alive():
            self._schedule_one(resolved, on_modified, on_deleted)

        logger.info("Registered file for watching: %s", resolved)

    def touch_debounce(self, file_path: Path) -> None:
        """Pre-arm the debounce timer for a path without registering it.

        Call this BEFORE modifying a watched file on disk (e.g. before
        shutil.copy2) so that the filesystem event triggered by the write
        is suppressed by the debounce check.
        """
        key = str(file_path.resolve())
        with self._debounce_lock:
            self._last_event_time[key] = time.time()

    def remove_file_watch(self, file_path: Path) -> bool:
        """Unregister a file.  Returns True if the file was being watched."""
        resolved = file_path.resolve()
        key = str(resolved)

        with self._reg_lock:
            if key not in self._registered:
                return False
            del self._registered[key]

        handle = self._watch_handles.pop(key, None)
        if handle is not None and self._observer is not None:
            try:
                self._observer.unschedule(handle)
            except Exception:
                pass

        logger.info("Unregistered file from watching: %s", resolved)
        return True

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start_background(self) -> None:
        """Start (or restart) the observer thread in the background.

        All registered files are (re-)scheduled on the new observer.
        """
        if self._observer is not None and self._observer.is_alive():
            logger.debug("Watcher already running.")
            return

        self._observer = Observer()
        # Mark as daemon so the thread never blocks process exit (e.g. on
        # Ctrl+C / SIGTERM the server shuts down without waiting for watchdog).
        self._observer.daemon = True
        self._watch_handles.clear()

        with self._reg_lock:
            snapshot = dict(self._registered)

        for key, (on_mod, on_del) in snapshot.items():
            self._schedule_one(Path(key), on_mod, on_del)

        self._observer.start()
        logger.info(
            "File watcher started (background) — watching %d file(s).",
            len(snapshot),
        )

    def stop(self) -> None:
        """Stop the observer thread.  The registered file list is preserved."""
        if self._observer is not None and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            logger.info(
                "File watcher stopped. %d file(s) remain registered.",
                len(self._registered),
            )
        self._observer = None
        self._watch_handles.clear()

    # ── Properties ────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._observer is not None and self._observer.is_alive()

    @property
    def registered_file_paths(self) -> list[Path]:
        with self._reg_lock:
            return [Path(k) for k in self._registered]

    def get_stats(self) -> dict:
        return {
            "is_running": self.is_running,
            "registered_file_count": len(self._registered),
            "registered_files": [str(p) for p in self.registered_file_paths],
        }

    # ── Internal ──────────────────────────────────────────────────────

    def _schedule_one(
        self,
        file_path: Path,
        on_modified: Callable[[Path], None],
        on_deleted: Callable[[Path], None] | None,
    ) -> None:
        """Schedule a single-file handler on the running observer."""
        resolved = file_path.resolve()
        key = str(resolved)
        watcher = self

        class _Handler(FileSystemEventHandler):
            def on_modified(self, event: FileSystemEvent) -> None:
                if (
                    not event.is_directory
                    and isinstance(event, FileModifiedEvent)
                    and Path(event.src_path).resolve() == resolved
                ):
                    watcher._fire(resolved, on_modified)

            def on_deleted(self, event: FileSystemEvent) -> None:
                if (
                    not event.is_directory
                    and isinstance(event, FileDeletedEvent)
                    and Path(event.src_path).resolve() == resolved
                    and on_deleted is not None
                ):
                    watcher._fire(resolved, on_deleted)

        handler = _Handler()
        try:
            watch_handle = self._observer.schedule(
                handler, str(resolved.parent), recursive=False
            )
            self._watch_handles[key] = watch_handle
        except Exception as e:
            logger.error("Failed to schedule watch for %s: %s", resolved, e)

    def _fire(self, file_path: Path, callback: Callable[[Path], None]) -> None:
        """Invoke callback with debouncing."""
        key = str(file_path)
        now = time.time()

        with self._debounce_lock:
            last = self._last_event_time.get(key, 0.0)
            if now - last < self.debounce_seconds:
                return
            self._last_event_time[key] = now

        logger.info("File changed, triggering incremental update: %s", file_path.name)
        try:
            callback(file_path)
        except Exception as e:
            logger.error(
                "Error during incremental update for %s: %s",
                file_path.name,
                e,
                exc_info=True,
            )
