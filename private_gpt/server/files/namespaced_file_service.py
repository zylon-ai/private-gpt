"""Namespace-aware file service for the Files API (T1.3).

Provides upload/write, read, delete, list-by-prefix, and stat operations
across any registered filesystem namespace.  Existing session-only callers
continue to work via the ``session`` namespace.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

import magic  # ty:ignore[unresolved-import]
from fastapi import HTTPException
from injector import inject, singleton

from private_gpt.components.filesystems.path_resolver import (
    InvalidPathError,
    PathEscapeError,
    PathResolver,
)
from private_gpt.server.files.file_models import (
    DeletedFile,
    FileListResponse,
    FileMetadata,
    FileScope,
)

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import UploadFile

logger = logging.getLogger(__name__)


def _detect_mime(content: bytes) -> str:
    try:
        return magic.Magic(mime=True).from_buffer(content)
    except Exception:
        return "application/octet-stream"


def _compute_etag(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()


@singleton
class NamespacedFileService:
    """Files API service that operates over named namespaces.

    Callers supply ``(namespace, scope, path)`` instead of raw filesystem
    paths.  The resolver enforces containment; the service reads/writes the
    underlying local filesystem.
    """

    @inject
    def __init__(self, resolver: PathResolver) -> None:
        self._resolver = resolver

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def write_file(
        self,
        namespace: str,
        scope: str,
        path: str,
        content: bytes,
        mime_type: str | None = None,
    ) -> FileMetadata:
        """Write *content* at ``(namespace, scope, path)``."""
        target = self._resolve(namespace, scope, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        mime = mime_type or _detect_mime(content)
        etag = _compute_etag(content)
        return _path_to_metadata(target, namespace, scope, path, mime, etag)

    async def upload_file(
        self,
        namespace: str,
        scope: str,
        path: str,
        upload: UploadFile,
    ) -> FileMetadata:
        """Stream-upload *upload* into the namespace."""
        content = await upload.read()
        mime = _detect_mime(content)
        return await self.write_file(namespace, scope, path, content, mime)

    async def read_file(self, namespace: str, scope: str, path: str) -> bytes:
        """Return the raw bytes of the file at ``(namespace, scope, path)``."""
        target = self._resolve(namespace, scope, path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {path!r}")
        return target.read_bytes()

    async def delete_file(self, namespace: str, scope: str, path: str) -> DeletedFile:
        """Delete the file at ``(namespace, scope, path)``."""
        target = self._resolve(namespace, scope, path)
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {path!r}")
        target.unlink()
        return DeletedFile(id=path)

    async def list_by_prefix(
        self,
        namespace: str,
        scope: str,
        prefix: str = "",
        limit: int = 100,
    ) -> FileListResponse:
        """List files under ``scope/prefix`` within the namespace."""
        root = self._resolve(namespace, scope, prefix)
        if not root.exists():
            return FileListResponse(data=[])

        ns_root = self._resolver.root(namespace) / scope
        files: list[FileMetadata] = []
        for entry in sorted(root.rglob("*")):
            if not entry.is_file():
                continue
            rel = str(entry.relative_to(ns_root))
            content = entry.read_bytes()
            mime = _detect_mime(content)
            etag = _compute_etag(content)
            files.append(_path_to_metadata(entry, namespace, scope, rel, mime, etag))

        page = files[:limit]
        return FileListResponse(
            data=page,
            first_id=page[0].id if page else None,
            last_id=page[-1].id if page else None,
            has_more=len(files) > limit,
        )

    async def stat_file(self, namespace: str, scope: str, path: str) -> FileMetadata:
        """Return metadata (including etag) for the file at the given location."""
        target = self._resolve(namespace, scope, path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {path!r}")
        content = target.read_bytes()
        mime = _detect_mime(content)
        etag = _compute_etag(content)
        return _path_to_metadata(target, namespace, scope, path, mime, etag)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, namespace: str, scope: str, path: str) -> Path:
        try:
            return self._resolver.resolve(namespace, scope, path)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (InvalidPathError, PathEscapeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


def _path_to_metadata(
    target: Path,
    namespace: str,
    scope: str,
    path: str,
    mime: str,
    etag: str,
) -> FileMetadata:
    from datetime import UTC, datetime

    stat = target.stat() if target.exists() else None
    return FileMetadata(
        id=path,
        created_at=(
            datetime.fromtimestamp(stat.st_mtime, tz=UTC)
            if stat
            else datetime.now(tz=UTC)
        ),
        filename=target.name,
        mime_type=mime,
        size_bytes=stat.st_size if stat else len(b""),
        downloadable=True,
        etag=etag,
        namespace=namespace,
        scope=FileScope(id=scope, type=namespace),
    )
