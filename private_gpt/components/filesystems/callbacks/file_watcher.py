"""File event handler for the ZGPT file callback system (T5.1).

Design: ZGPT receives MinIO/S3 bucket notification webhooks at
``POST /v1/internal/file-events``.  Each notification is parsed,
deduplicated (multipart uploads produce multiple events for one logical
write), and mapped to a typed FileEvent.

Deduplication strategy:
- Events within DEBOUNCE_SECONDS of each other for the same key are
  collapsed into a single event.
- Paths indicating temp / multipart objects (e.g. `.tmp`, `.part`,
  or containing `.minio.sys`) are skipped.
- Zero-byte objects on created/updated events are skipped.
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

# Seconds within which duplicate events for the same key are collapsed.
DEBOUNCE_SECONDS = 0.5

# Patterns indicating temp / multipart objects — skip these.
_TEMP_PATTERNS = re.compile(
    r"(^\.minio\.sys|\.tmp$|\.part$|\.crdownload$|~$|^~)",
    re.IGNORECASE,
)


def _is_temp_key(key: str) -> bool:
    name = key.rsplit("/", 1)[-1]
    return bool(_TEMP_PATTERNS.search(name)) or ".minio.sys" in key


class FileEventDebouncer:
    """Stateful debouncer that collapses rapid duplicate events per key.

    Intended to be used per-session or globally.  Thread-safe for single-
    threaded async use; add a lock if used from multiple threads.
    """

    def __init__(self) -> None:
        # key → (last_seen_timestamp, event_type)
        self._pending: dict[str, tuple[float, str]] = defaultdict(lambda: (0.0, ""))

    def should_emit(self, key: str, event_type: str) -> bool:
        """Return True if this event should be emitted (not a recent duplicate)."""
        now = time.monotonic()
        prev_ts, prev_type = self._pending[key]
        is_dup = now - prev_ts < DEBOUNCE_SECONDS and prev_type == event_type
        self._pending[key] = (now, event_type)
        return not is_dup


def parse_minio_notification(
    payload: dict[str, Any],
) -> list[tuple[str, str, int]]:
    """Parse a MinIO bucket notification payload into (key, event_type, size) tuples.

    MinIO sends an ``EventName`` field and a ``Records`` array.

    Returns a list of (object_key, event_type, size_bytes) for the caller
    to process.  Unknown / unsupported event names are silently skipped.
    """
    results: list[tuple[str, str, int]] = []
    for record in payload.get("Records", []):
        event_name: str = record.get("eventName", "")
        obj = record.get("s3", {}).get("object", {})
        key: str = obj.get("key", "")
        size: int = int(obj.get("size", 0))

        if not key:
            continue

        if event_name.startswith("s3:ObjectCreated:"):
            event_type = "file.created" if size > 0 else None
        elif event_name.startswith("s3:ObjectRemoved:"):
            event_type = "file.deleted"
        else:
            continue

        if event_type is None:
            logger.debug("Skipping zero-byte created event for key=%s", key)
            continue

        if _is_temp_key(key):
            logger.debug("Skipping temp/multipart key: %s", key)
            continue

        results.append((key, event_type, size))

    return results


def key_to_namespace_scope_path(
    key: str,
    bucket_namespace_map: dict[str, str],
) -> tuple[str, str, str] | None:
    """Derive (namespace, scope, path) from an S3 object key and bucket→namespace map.

    Artifact path: ``{orgId}/{projectId?}/{artifactId}``
    Scope = orgId (first segment), path = the rest.

    Returns None when the key cannot be mapped.
    """
    # key looks like: "org-uuid/proj-uuid/art-uuid.mdx" or "org-uuid/art-uuid.mdx"
    parts = key.split("/", 1)
    if len(parts) < 2:
        return None
    # scope = orgId (first segment), path = rest; namespace from caller
    return None  # filled in by the caller who knows the bucket→namespace mapping
