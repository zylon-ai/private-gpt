"""File callback service for the ZGPT filesystem platform (T5.1).

Receives MinIO/S3 bucket notification webhooks, deduplicates them, and
routes typed FileEvents to configured callback targets (AMQP or HTTP).

Architecture:
  MinIO bucket webhook
    → POST /v1/internal/file-events   (registered in zylon-gpt layer)
    → FileCallbackService.handle_bucket_notification()
    → deduplication via FileEventDebouncer
    → typed FileEvent emitted to AMQP / HTTP callback targets

The callback targets are registered per-session by the chat layer (the
Backend tells ZGPT which exchange/routing-key to use for this turn via
the ``file_callbacks`` field in the ChatBody).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp
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

        self._sessions[session_id] = (namespace, target)
        if namespace not in self._debouncers:
            self._debouncers[namespace] = FileEventDebouncer()
        logger.info(
            "File watch registered: session=%s namespace=%s", session_id, namespace
        )

    def unregister_watch(self, session_id: str) -> None:
        """Unregister callbacks for a session."""
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
        """
        events = parse_minio_notification(payload)
        if not events:
            return

        debouncer = self._debouncers.get(namespace)
        if debouncer is None:
            return  # no sessions watching this namespace

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
                        update={"correlation": {**target.correlation, **ev.correlation}}
                    )
                    _task = asyncio.create_task(self._dispatch(correlated_ev, target))  # noqa: RUF006

    async def _dispatch(self, event: FileEvent, target: FileCallbackTarget) -> None:
        try:
            if target.amqp:
                await self._dispatch_amqp(event, target)
            if target.http:
                await self._dispatch_http(event, target)
        except Exception as exc:
            logger.warning("Failed to dispatch file event %s: %s", event.type, exc)

    async def _dispatch_amqp(
        self, event: FileEvent, target: FileCallbackTarget
    ) -> None:
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

    async def _dispatch_http(
        self, event: FileEvent, target: FileCallbackTarget
    ) -> None:
        if not target.http:
            return
        payload = event.model_dump(mode="json")
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                target.http.url,
                json=payload,
                headers={"Content-Type": "application/json", **target.http.headers},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp,
        ):
            if resp.status >= 400:
                logger.warning(
                    "File callback HTTP error: status=%s url=%s",
                    resp.status,
                    target.http.url,
                )
