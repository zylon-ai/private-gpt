"""File callback service for the ZGPT filesystem platform.

Receives MinIO/S3 bucket notification webhooks, deduplicates them, and
routes typed FileEvents to configured callback targets (AMQP or HTTP).

Architecture:
  MinIO bucket webhook
    → POST /v1/internal/file-events   (registered in zylon-gpt layer)
    → FileCallbackService.handle_bucket_notification()
    → deduplication via FileEventDebouncer
    → typed FileEvent enqueued on a background delivery thread
    → delivered to AMQP / HTTP callback targets

Delivery follows the same pattern as the broker's ``BlockingPublisher``:
the caller only enqueues work and returns immediately, while the blocking
HTTP delivery happens on a dedicated daemon thread.  The delivery thread is
only started in the API process — worker processes (identified by
``PGPT_WORKER_MODE`` being set) never run it, because file events are only
meaningful where the webhook receiver and session registrations live.

The callback targets are registered per-session by the chat layer (the
Backend tells ZGPT which exchange/routing-key to use for this turn via
the ``file_callbacks`` field in the ChatBody).
"""

from __future__ import annotations

import atexit
import logging
import os
import queue
import threading
import time
from typing import Any, NamedTuple

import requests
from injector import inject, singleton

from private_gpt.components.filesystems.callbacks.file_watcher import (
    FileEventDebouncer,
    parse_minio_notification,
)
from private_gpt.components.filesystems.callbacks.models import (
    FileCallbackTarget,
    FileEvent,
)
from private_gpt.components.filesystems.namespace_registry import NamespaceRegistry

logger = logging.getLogger(__name__)

# Timeout for a single HTTP callback delivery.
HTTP_CALLBACK_TIMEOUT_SECONDS = 5

# Sentinel pushed on close to wake the delivery loop.
_STOP = None


class CallbackJob(NamedTuple):
    """A file event queued for delivery to a callback target."""

    event: FileEvent
    target: FileCallbackTarget


def _is_worker_process() -> bool:
    """Return True when running inside a private-gpt worker process.

    Worker processes are launched via ``private-gpt worker`` with
    ``PGPT_WORKER_MODE`` set (see ``private_gpt.worker``).  The API process
    never sets it, so it cleanly discriminates the two runtimes.
    """
    return bool(os.environ.get("PGPT_WORKER_MODE", "").strip())


class _CallbackWorker(threading.Thread):
    """Background thread that delivers file events to callback targets.

    Callers enqueue jobs and return immediately (mirroring
    ``BlockingPublisher``); this thread owns the blocking HTTP delivery so a
    webhook handler or chat turn never blocks on callback I/O.
    """

    def __init__(self) -> None:
        super().__init__(name="FileCallbackWorker", daemon=True)
        self._queue: queue.Queue[CallbackJob | None] = queue.Queue()

    def enqueue(self, job: CallbackJob) -> None:
        """Queue a file event for delivery. Never blocks."""
        self._queue.put(job)

    def drain(self, timeout: float = 5.0) -> None:
        """Block until all queued jobs have been delivered."""
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks > 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        if self._queue.unfinished_tasks > 0:
            logger.warning("Timed out waiting for file callback delivery")

    def close(self, timeout: float = 5.0) -> None:
        """Stop the worker, draining any jobs still queued.

        The sentinel is pushed last, so every job already queued is delivered
        before the loop exits.
        """
        self._queue.put(_STOP)
        self.join(timeout=timeout)

    def run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                break
            try:
                self._deliver(job)
            except Exception:
                logger.exception("Failed to deliver file event %s", job.event.type)
            finally:
                self._queue.task_done()

    def _deliver(self, job: CallbackJob) -> None:
        event, target = job
        if target.amqp:
            self._deliver_amqp(event, target)
        if target.http:
            self._deliver_http(event, target)

    def _deliver_amqp(self, event: FileEvent, target: FileCallbackTarget) -> None:
        if not target.amqp:
            return
        # Full AMQP dispatch is wired in the zylon-gpt layer which has aio_pika.
        # Log here so it's visible and testable.
        logger.info(
            "FileEvent[AMQP] exchange=%s routing_key=%s type=%s path=%s",
            target.amqp.exchange,
            target.amqp.routing_key,
            event.type,
            event.path,
        )

    def _deliver_http(self, event: FileEvent, target: FileCallbackTarget) -> None:
        if not target.http:
            return
        payload = event.model_dump(mode="json")
        try:
            response = requests.post(
                target.http.url,
                json=payload,
                headers={"Content-Type": "application/json", **target.http.headers},
                timeout=HTTP_CALLBACK_TIMEOUT_SECONDS,
            )
        except requests.RequestException:
            logger.warning(
                "File callback HTTP request failed: url=%s",
                target.http.url,
                exc_info=True,
            )
            return
        if response.status_code >= 400:
            logger.warning(
                "File callback HTTP error: status=%s url=%s",
                response.status_code,
                target.http.url,
            )


@singleton
class FileCallbackService:
    """Routes MinIO bucket notifications to typed file events.

    Usage:
        service.register_watch(session_id, namespace, target)
        # MinIO sends webhook → handle_bucket_notification(bucket, payload)
        service.unregister_watch(session_id)
    """

    @inject
    def __init__(self, registry: NamespaceRegistry) -> None:
        self._registry = registry
        # session_id → (namespace, FileCallbackTarget)
        self._sessions: dict[str, tuple[str, FileCallbackTarget]] = {}
        # namespace → FileEventDebouncer (shared per namespace)
        self._debouncers: dict[str, FileEventDebouncer] = {}
        # bucket → namespace (populated from registry at startup)
        self._bucket_namespace_map: dict[str, str] = {}
        self._lock = threading.RLock()

        # Callback delivery runs on a background thread, like the broker
        # publisher. Only the API process starts it: worker processes set
        # PGPT_WORKER_MODE and never receive file events.
        self._worker: _CallbackWorker | None = None
        if not _is_worker_process():
            self._worker = _CallbackWorker()
            self._worker.start()
            atexit.register(self.close)

    def register_watch(
        self,
        session_id: str,
        namespace: str,
        target: FileCallbackTarget,
    ) -> None:
        """Register a callback target for a session's namespace."""
        try:
            self._registry.get(namespace)
        except KeyError:
            logger.debug(
                "Namespace '%s' not registered — skipping watch for session %s",
                namespace,
                session_id,
            )
            return

        with self._lock:
            self._sessions[session_id] = (namespace, target)
            if namespace not in self._debouncers:
                self._debouncers[namespace] = FileEventDebouncer()
        logger.info(
            "File watch registered: session=%s namespace=%s", session_id, namespace
        )

    def unregister_watch(self, session_id: str) -> None:
        """Unregister callbacks for a session."""
        with self._lock:
            self._sessions.pop(session_id, None)
        logger.debug("File watch unregistered: session=%s", session_id)

    async def handle_bucket_notification(
        self,
        namespace: str,
        payload: dict[str, Any],
        scope_hint: str | None = None,
    ) -> None:
        """Process an incoming MinIO/S3 bucket notification for a namespace.

        The ``scope_hint`` is an optional orgId/scope taken from the caller
        (e.g. from an authenticated webhook with org context).

        Parsing and deduplication happen inline; the actual callback delivery
        is enqueued on the background delivery thread (API process only).
        """
        events = parse_minio_notification(payload)
        if not events:
            return

        with self._lock:
            debouncer = self._debouncers.get(namespace)
            if debouncer is None:
                return  # no sessions watching this namespace

            worker = self._worker
            if worker is None:
                logger.debug(
                    "File callback worker not running (worker process) — "
                    "dropping %s event(s) for namespace=%s",
                    len(events),
                    namespace,
                )
                return

            for key, event_type, _size in events:
                # Derive scope (orgId = first path segment) and path from key
                parts = key.split("/", 1)
                scope = parts[0] if len(parts) >= 2 else (scope_hint or "")
                path = parts[1] if len(parts) >= 2 else key

                if not debouncer.should_emit(key, event_type):
                    logger.debug(
                        "Debounced duplicate event: key=%s type=%s", key, event_type
                    )
                    continue

                ev = FileEvent(
                    type=event_type,  # type: ignore[arg-type]
                    path=path,
                    namespace=namespace,
                    scope=scope,
                )

                # Dispatch to all sessions watching this namespace
                for _session_id, (sess_ns, target) in list(self._sessions.items()):
                    if sess_ns == namespace:
                        # Merge session correlation into event
                        correlated_ev = ev.model_copy(
                            update={
                                "correlation": {**target.correlation, **ev.correlation}
                            }
                        )
                        worker.enqueue(CallbackJob(event=correlated_ev, target=target))

    def drain(self, timeout: float = 5.0) -> None:
        """Block until all queued file events have been delivered.

        No-op when the delivery worker is not running (worker process).
        """
        if self._worker is not None:
            self._worker.drain(timeout=timeout)

    def close(self) -> None:
        """Stop the delivery worker thread, draining queued jobs."""
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.close()
