"""File callback service for the ZGPT filesystem platform (T5.1).

Manages per-session watch sessions and routes FileEvents to configured
callback targets (AMQP or HTTP).
"""

from __future__ import annotations

import asyncio
import logging

import aiohttp
from injector import inject, singleton

from private_gpt.components.filesystems.callbacks.file_watcher import FileWatchSession
from private_gpt.components.filesystems.callbacks.models import (
    FileCallbackTarget,
    FileEvent,
)
from private_gpt.components.filesystems.namespace_registry import NamespaceRegistry

logger = logging.getLogger(__name__)


@singleton
class FileCallbackService:
    """Manages filesystem watch sessions and emits typed file events.

    Usage:
        service.start_watch(session_id, namespace, scope, target)
        # ... sandbox runs ...
        service.stop_watch(session_id)
    """

    @inject
    def __init__(self, registry: NamespaceRegistry) -> None:
        self._registry = registry
        self._sessions: dict[str, FileWatchSession] = {}

    def start_watch(
        self,
        session_id: str,
        namespace: str,
        scope: str,
        target: FileCallbackTarget,
    ) -> None:
        """Start watching namespace/scope for a session.

        Skips silently when the namespace is not registered (optional mount).
        """
        try:
            ns_root = self._registry.root(namespace)
        except KeyError:
            logger.debug(
                "Namespace '%s' not registered — skipping watch for session %s",
                namespace,
                session_id,
            )
            return

        watch_path = ns_root / scope
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        session = FileWatchSession(
            namespace=namespace,
            scope=scope,
            watch_path=watch_path,
            correlation=target.correlation,
            emit=lambda ev: self._dispatch(ev, target),
            loop=loop,
        )
        session.start()
        self._sessions[session_id] = session
        logger.info(
            "File watch started: session=%s namespace=%s scope=%s",
            session_id,
            namespace,
            scope,
        )

    def stop_watch(self, session_id: str) -> None:
        """Stop watching for a session. No-op if not watching."""
        session = self._sessions.pop(session_id, None)
        if session:
            session.stop()
            logger.info("File watch stopped: session=%s", session_id)

    async def _dispatch(self, event: FileEvent, target: FileCallbackTarget) -> None:
        """Route a FileEvent to all configured targets."""
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
        """Publish event to AMQP. Requires aio_pika to be wired via DI."""
        if not target.amqp:
            return
        # aio_pika wiring is done in the zylon-gpt layer which has the broker.
        # For now, log the event so it can be consumed by any log-based consumer.
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
        """POST event payload to the HTTP callback URL."""
        if not target.http:
            return
        payload = event.model_dump(mode="json")
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                target.http.url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    **target.http.headers,
                },
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp,
        ):
            if resp.status >= 400:
                logger.warning(
                    "File callback HTTP error: status=%s url=%s",
                    resp.status,
                    target.http.url,
                )
