from __future__ import annotations

import base64
import hashlib
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import magic  # ty:ignore[unresolved-import]
from fastapi import HTTPException
from injector import inject, singleton

from private_gpt.components.environment.layout import (
    canonical_to_storage_path,
    storage_to_canonical_path,
)
from private_gpt.components.filesystems.namespace_registry import NamespaceRegistry
from private_gpt.components.filesystems.path_resolver import (
    InvalidPathError,
    PathEscapeError,
    PathResolver,
)
from private_gpt.components.storage.storage_component import StorageComponent
from private_gpt.server.files.file_models import (
    DeletedFile,
    FileListResponse,
    FileMetadata,
    FileScope,
    NamespaceInfo,
    NamespaceListResponse,
)
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from fastapi import UploadFile

    from private_gpt.components.storage.models import FileInfo
    from private_gpt.components.storage.object_storage import ObjectStorage


def _detect_mime_from_bytes(content: bytes) -> str:
    try:
        return magic.Magic(mime=True).from_buffer(content)
    except Exception:
        return "application/octet-stream"


def _compute_etag(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()


def _encode_file_id(path: str) -> str:
    return base64.urlsafe_b64encode(path.encode()).decode().rstrip("=")


def _decode_file_id(file_id: str) -> str:
    padding = (4 - len(file_id) % 4) % 4
    try:
        return base64.urlsafe_b64decode(file_id + "=" * padding).decode()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid file ID encoding.") from e


@singleton
class FileService:
    @inject
    def __init__(
        self,
        storage_component: StorageComponent,
        settings: Settings,
        registry: NamespaceRegistry,
        resolver: PathResolver,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._resolver = resolver
        cfg = settings.code_execution
        # Find the namespace that uses the ObjectStorage backend (normally "session").
        storage_ns = next(
            (n for n in registry.all_names() if registry.get(n).storage_backend),
            None,
        )
        if storage_ns:
            local_root = str(registry.root(storage_ns))
        else:
            local_root = str(Path(settings.data.local_data_folder) / "code_execution")
        self._storage = storage_component.get_object_storage(
            provider=cfg.storage_provider,
            local_root_path=local_root,
            bucket_name=settings.s3.durable_bucket_name,
        )

    def list_namespaces(self) -> NamespaceListResponse:
        """Return all registered filesystem namespaces, sorted by name."""
        names = sorted(self._registry.all_names())
        return NamespaceListResponse(
            data=[
                NamespaceInfo(
                    name=name,
                    root=str(self._registry.root(name)),
                    default_mode=self._registry.get(name).default_mode,
                )
                for name in names
            ]
        )

    def _ns_uses_storage(self, namespace: str) -> bool:
        """Return True when *namespace* is served by the ObjectStorage backend.

        Namespaces with ``storage_backend=True`` expose the uploads/ and outputs/
        virtual-folder layout via ObjectStorage.  All other namespaces are served
        directly from the local filesystem via PathResolver.
        """
        try:
            return self._registry.get(namespace).storage_backend
        except KeyError:
            return False

    def _require_storage(self) -> ObjectStorage:
        return self._storage

    def _uploads_prefix(self, scope_id: str) -> str:
        return f"uploads/{scope_id}"

    def _outputs_prefix(self, scope_id: str) -> str:
        return f"outputs/{scope_id}"

    def _prefix_for_path(self, storage_path: str, scope_id: str) -> str:
        """Return the storage prefix for a given storage_path."""
        folder = storage_path.split("/")[0]
        return f"{folder}/{scope_id}"

    def _to_metadata(
        self,
        file_info: FileInfo,
        scope_id: str,
        namespace: str = "session",
    ) -> FileMetadata:
        downloadable = not file_info.path.startswith("uploads/")
        canonical = storage_to_canonical_path(file_info.path)
        return FileMetadata(
            id=_encode_file_id(canonical),
            created_at=file_info.created_at,
            filename=file_info.path.split("/")[-1],
            mime_type=file_info.mime_type,
            size_bytes=file_info.size_bytes,
            downloadable=downloadable,
            etag=file_info.etag,
            namespace=namespace,
            scope=FileScope(id=scope_id, type=namespace),
        )

    async def upload_file(
        self, scope_id: str, upload: UploadFile, path: str | None = None
    ) -> FileMetadata:
        storage = self._require_storage()
        content = await upload.read()
        filename = upload.filename or "upload"
        mime_type = _detect_mime_from_bytes(content)

        rel_path = self._normalize_upload_path(path=path, fallback=filename)
        prefix = self._uploads_prefix(scope_id)
        await storage.write_file(prefix, rel_path, content, mime_type)

        file_info = await storage.stat_file(prefix, rel_path)
        if file_info is None:
            raise HTTPException(
                status_code=500, detail="File written but could not be read back."
            )
        file_info = file_info.model_copy(update={"path": f"uploads/{rel_path}"})
        return self._to_metadata(file_info, scope_id)

    async def put_file(
        self,
        scope_id: str,
        path: str,
        content: bytes,
        mime_type: str | None = None,
    ) -> FileMetadata:
        """S3/blob-style put-object: store *content* at an arbitrary key.

        The key is interpreted relative to the session's uploads mount (i.e.
        ``/mnt/user-data/uploads/`` in the sandbox), so files pushed this way
        stay visible to the agent and to the existing listing/download flows.
        """
        storage = self._require_storage()
        rel_path = self._normalize_upload_path(path=path, fallback="upload")
        prefix = self._uploads_prefix(scope_id)
        await storage.write_file(
            prefix, rel_path, content, mime_type or _detect_mime_from_bytes(content)
        )

        file_info = await storage.stat_file(prefix, rel_path)
        if file_info is None:
            raise HTTPException(
                status_code=500, detail="File written but could not be read back."
            )
        file_info = file_info.model_copy(update={"path": f"uploads/{rel_path}"})
        return self._to_metadata(file_info, scope_id)

    async def list_files(
        self,
        scope_id: str,
        limit: int = 20,
        after_id: str | None = None,
        before_id: str | None = None,
        namespace: str = "session",
        prefix: str | None = None,
    ) -> FileListResponse:
        """List files, optionally filtered by a canonical path prefix.

        The *prefix* is matched against the canonical upload key (relative to
        the uploads mount, e.g. ``data/2024/``).  For the session namespace
        both the ``uploads/`` and ``outputs/`` folder keys are checked against
        the prefix.  An empty or absent prefix returns everything.
        """
        if not self._ns_uses_storage(namespace):
            return await self._list_namespace_files(scope_id, namespace, limit, prefix)

        storage = self._require_storage()
        norm_prefix = (prefix or "").strip().lstrip("/")

        uploads = await storage.list_files_meta(self._uploads_prefix(scope_id))
        outputs = await storage.list_files_meta(self._outputs_prefix(scope_id))

        def _keep(storage_rel_path: str) -> bool:
            if not norm_prefix:
                return True
            return storage_rel_path.startswith(norm_prefix)

        all_infos = sorted(
            [
                *[
                    fi.model_copy(update={"path": f"uploads/{fi.path}"})
                    for fi in uploads
                    if _keep(fi.path)
                ],
                *[
                    fi.model_copy(update={"path": f"outputs/{fi.path}"})
                    for fi in outputs
                    if _keep(fi.path)
                ],
            ],
            key=lambda fi: fi.created_at,
        )
        all_files = [self._to_metadata(fi, scope_id, namespace) for fi in all_infos]

        if after_id:
            ids = [f.id for f in all_files]
            try:
                idx = ids.index(after_id)
                all_files = all_files[idx + 1 :]
            except ValueError:
                pass

        if before_id:
            ids = [f.id for f in all_files]
            try:
                idx = ids.index(before_id)
                all_files = all_files[:idx]
            except ValueError:
                pass

        has_more = len(all_files) > limit
        page = all_files[:limit]

        return FileListResponse(
            data=page,
            first_id=page[0].id if page else None,
            last_id=page[-1].id if page else None,
            has_more=has_more,
        )

    async def get_file_metadata(
        self,
        scope_id: str,
        file_id: str,
        namespace: str = "session",
    ) -> FileMetadata:
        if not self._ns_uses_storage(namespace):
            return await self._stat_namespace_file(scope_id, namespace, file_id)

        storage = self._require_storage()
        canonical = _decode_file_id(file_id)
        storage_path = canonical_to_storage_path(canonical)
        self._validate_file_id(storage_path)
        folder, filename = storage_path.split("/", 1)
        prefix = self._prefix_for_path(storage_path, scope_id)
        file_info = await storage.stat_file(prefix, filename)
        if file_info is None:
            raise HTTPException(status_code=404, detail=f"File '{file_id}' not found.")
        file_info = file_info.model_copy(update={"path": f"{folder}/{filename}"})
        return self._to_metadata(file_info, scope_id, namespace)

    async def get_file_content(
        self,
        scope_id: str,
        file_id: str,
        namespace: str = "session",
    ) -> tuple[bytes, str, str]:
        """Returns (bytes, mime_type, display_filename)."""
        if not self._ns_uses_storage(namespace):
            return await self._read_namespace_file(scope_id, namespace, file_id)

        storage = self._require_storage()
        canonical = _decode_file_id(file_id)
        storage_path = canonical_to_storage_path(canonical)
        self._validate_file_id(storage_path)
        _folder, filename = storage_path.split("/", 1)
        prefix = self._prefix_for_path(storage_path, scope_id)
        file_info = await storage.stat_file(prefix, filename)
        if file_info is None:
            raise HTTPException(status_code=404, detail=f"File '{file_id}' not found.")
        content = await storage.read_file(prefix, filename)
        display_name = canonical.split("/")[-1]
        return content, file_info.mime_type, display_name

    async def delete_file(
        self,
        scope_id: str,
        file_id: str,
        namespace: str = "session",
    ) -> DeletedFile:
        if not self._ns_uses_storage(namespace):
            return await self._delete_namespace_file(scope_id, namespace, file_id)

        storage = self._require_storage()
        canonical = _decode_file_id(file_id)
        storage_path = canonical_to_storage_path(canonical)
        self._validate_file_id(storage_path)
        if storage_path.startswith("user/"):
            prefix = f"user/{scope_id}"
            _folder, filename = storage_path.split("/", 1)
        elif storage_path.startswith("uploads/"):
            prefix = self._uploads_prefix(scope_id)
            _folder, filename = storage_path.split("/", 1)
        elif storage_path.startswith("outputs/"):
            prefix = self._outputs_prefix(scope_id)
            _folder, filename = storage_path.split("/", 1)
        else:
            raise HTTPException(
                status_code=404,
                detail=f"File '{file_id}' not found.",
            )

        deleted = await storage.delete_file(prefix, filename)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"File '{file_id}' not found.",
            )
        return DeletedFile(id=file_id)

    async def delete_prefix(
        self,
        scope_id: str,
        prefix: str,
        namespace: str = "session",
    ) -> int:
        """Delete every file whose upload key starts with *prefix*.

        Returns the number of files actually deleted.  Only upload-side files
        are affected (output files cannot be bulk-deleted through the API).
        Works for both session storage (via ObjectStorage) and other namespaces
        (via local PathResolver).
        """
        if not self._ns_uses_storage(namespace):
            return await self._delete_namespace_prefix(scope_id, namespace, prefix)

        norm = (prefix or "").strip().lstrip("/")
        if not norm:
            raise HTTPException(
                status_code=400,
                detail="prefix must be non-empty for bulk delete.",
            )
        storage = self._require_storage()
        uploads = await storage.list_files_meta(self._uploads_prefix(scope_id))
        targets = [fi.path for fi in uploads if fi.path.startswith(norm)]
        count = 0
        for rel_path in targets:
            deleted = await storage.delete_file(
                self._uploads_prefix(scope_id), rel_path
            )
            if deleted:
                count += 1
        return count

    async def head_file(
        self,
        scope_id: str,
        file_id: str,
        namespace: str = "session",
    ) -> FileMetadata:
        """Lightweight stat that avoids reading file bytes where possible.

        For the session namespace this delegates to the underlying
        ``stat_file`` on the ObjectStorage (which only reads metadata, not
        content, on S3-compatible backends).  For namespace-backed paths it
        falls back to the regular stat (local FS stat is always O(1)).
        """
        return await self.get_file_metadata(scope_id, file_id, namespace)

    # ------------------------------------------------------------------
    # Namespace-aware operations via PathResolver (local FS)
    # ------------------------------------------------------------------

    def _resolve_ns_path(self, namespace: str, scope: str, path: str) -> Path:
        try:
            return self._resolver.resolve(namespace, scope, path)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (InvalidPathError, PathEscapeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def write_file_to_namespace(
        self,
        namespace: str,
        scope: str,
        path: str,
        content: bytes,
        mime_type: str | None = None,
    ) -> FileMetadata:
        """Write bytes at (namespace, scope, path) on the local FS."""
        target = self._resolve_ns_path(namespace, scope, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        mime = mime_type or _detect_mime_from_bytes(content)
        etag = _compute_etag(content)
        from datetime import UTC, datetime

        stat = target.stat()
        return FileMetadata(
            id=path,
            created_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            filename=target.name,
            mime_type=mime,
            size_bytes=stat.st_size,
            downloadable=True,
            etag=etag,
            namespace=namespace,
            scope=FileScope(id=scope, type=namespace),
        )

    async def _stat_namespace_file(
        self, scope: str, namespace: str, path: str
    ) -> FileMetadata:
        target = self._resolve_ns_path(namespace, scope, path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {path!r}")
        content = target.read_bytes()
        mime = _detect_mime_from_bytes(content)
        etag = _compute_etag(content)
        from datetime import UTC, datetime

        stat = target.stat()
        return FileMetadata(
            id=path,
            created_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            filename=target.name,
            mime_type=mime,
            size_bytes=stat.st_size,
            downloadable=True,
            etag=etag,
            namespace=namespace,
            scope=FileScope(id=scope, type=namespace),
        )

    async def _read_namespace_file(
        self, scope: str, namespace: str, path: str
    ) -> tuple[bytes, str, str]:
        target = self._resolve_ns_path(namespace, scope, path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {path!r}")
        content = target.read_bytes()
        mime = _detect_mime_from_bytes(content)
        return content, mime, target.name

    async def _list_namespace_files(
        self, scope: str, namespace: str, limit: int = 100, prefix: str | None = None
    ) -> FileListResponse:
        ns_root = self._resolve_ns_path(namespace, scope, "")
        if not ns_root.exists():
            return FileListResponse(data=[])
        from datetime import UTC, datetime

        norm_prefix = (prefix or "").strip().lstrip("/")
        files: list[FileMetadata] = []
        for entry in sorted(ns_root.rglob("*")):
            if not entry.is_file():
                continue
            rel = str(entry.relative_to(ns_root))
            if norm_prefix and not rel.startswith(norm_prefix):
                continue
            content = entry.read_bytes()
            mime = _detect_mime_from_bytes(content)
            etag = _compute_etag(content)
            stat = entry.stat()
            files.append(
                FileMetadata(
                    id=rel,
                    created_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                    filename=entry.name,
                    mime_type=mime,
                    size_bytes=stat.st_size,
                    downloadable=True,
                    etag=etag,
                    namespace=namespace,
                    scope=FileScope(id=scope, type=namespace),
                )
            )

        page = files[:limit]
        return FileListResponse(
            data=page,
            first_id=page[0].id if page else None,
            last_id=page[-1].id if page else None,
            has_more=len(files) > limit,
        )

    async def _delete_namespace_file(
        self, scope: str, namespace: str, path: str
    ) -> DeletedFile:
        target = self._resolve_ns_path(namespace, scope, path)
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {path!r}")
        target.unlink()
        return DeletedFile(id=path)

    async def _delete_namespace_prefix(
        self, scope: str, namespace: str, prefix: str
    ) -> int:
        norm = (prefix or "").strip().lstrip("/")
        if not norm:
            raise HTTPException(
                status_code=400,
                detail="prefix must be non-empty for bulk delete.",
            )
        ns_root = self._resolve_ns_path(namespace, scope, "")
        if not ns_root.exists():
            return 0
        count = 0
        for entry in list(ns_root.rglob("*")):
            if not entry.is_file():
                continue
            rel = str(entry.relative_to(ns_root))
            if rel.startswith(norm):
                entry.unlink()
                count += 1
        return count

    @staticmethod
    def _validate_file_id(file_id: str) -> None:
        if ".." in file_id.split("/"):
            raise HTTPException(status_code=400, detail="Invalid file ID.")

    @staticmethod
    def _normalize_upload_path(path: str | None, fallback: str) -> str:
        """Normalize and validate an object-storage-style upload key.

        Keys are interpreted relative to the session's uploads mount. An
        explicit ``uploads/`` prefix is accepted and stripped so callers can
        pass the full storage key; ``../``, absolute paths and trailing
        slashes are rejected (mirrors S3/blob put-object key rules).
        """
        rel = (path or "").strip()
        if not rel:
            rel = fallback
        if rel.startswith("/"):
            raise HTTPException(
                status_code=400,
                detail="path must be relative (no leading '/').",
            )
        if rel.endswith("/"):
            raise HTTPException(
                status_code=400,
                detail="path must point to a file, not a directory (no trailing '/').",
            )
        parts = PurePosixPath(rel).parts
        if ".." in parts:
            raise HTTPException(
                status_code=400,
                detail="path must not contain '..' components.",
            )
        if parts and parts[0] == "uploads":
            rel = "/".join(parts[1:])
        if not rel:
            raise HTTPException(
                status_code=400,
                detail="path must not be empty.",
            )
        return rel
