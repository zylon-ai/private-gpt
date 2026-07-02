from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING

import magic
from fastapi import HTTPException
from injector import inject, singleton

from private_gpt.components.environment.layout import DEFAULT_SESSION_LAYOUT
from private_gpt.components.storage.storage_component import StorageComponent
from private_gpt.server.files.file_models import (
    DeletedFile,
    FileListResponse,
    FileMetadata,
    FileScope,
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


def _encode_file_id(path: str) -> str:
    return base64.urlsafe_b64encode(path.encode()).decode().rstrip("=")


def _decode_file_id(file_id: str) -> str:
    padding = (4 - len(file_id) % 4) % 4
    try:
        return base64.urlsafe_b64decode(file_id + "=" * padding).decode()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid file ID encoding.") from e


def _canonical_to_storage_path(canonical: str) -> str:
    for mount in DEFAULT_SESSION_LAYOUT:
        if canonical.startswith(mount.canonical):
            relative = canonical[len(mount.canonical) :]
            return f"{mount.name}/{relative}"
    return canonical


def _storage_to_canonical_path(storage: str) -> str:
    for mount in DEFAULT_SESSION_LAYOUT:
        prefix = f"{mount.name}/"
        if storage.startswith(prefix):
            relative = storage[len(prefix) :]
            return f"{mount.canonical}{relative}"
    return storage


@singleton
class FileService:
    @inject
    def __init__(self, storage_component: StorageComponent, settings: Settings) -> None:
        self._settings = settings
        cfg = settings.code_execution
        local_root = cfg.volume_root or str(
            Path(settings.data.local_data_folder) / "code_execution"
        )
        self._storage = storage_component.get_object_storage(
            provider=cfg.storage_provider,
            local_root_path=local_root,
            bucket_name=settings.s3.durable_bucket_name,
        )

    def _require_storage(self) -> ObjectStorage:
        return self._storage

    def _uploads_prefix(self, scope_id: str) -> str:
        return f"uploads/{scope_id}"

    def _outputs_prefix(self, scope_id: str) -> str:
        return f"outputs/{scope_id}"

    def _prefix_for_path(self, storage_path: str, scope_id: str) -> str:
        """Return the storage prefix for a given storage_path (e.g. 'uploads/file.csv')."""
        folder = storage_path.split("/")[0]
        return f"{folder}/{scope_id}"

    def _to_metadata(self, file_info: FileInfo, scope_id: str) -> FileMetadata:
        downloadable = not file_info.path.startswith("uploads/")
        canonical = _storage_to_canonical_path(file_info.path)
        return FileMetadata(
            id=_encode_file_id(canonical),
            created_at=file_info.created_at,
            filename=file_info.path.split("/")[-1],
            mime_type=file_info.mime_type,
            size_bytes=file_info.size_bytes,
            downloadable=downloadable,
            scope=FileScope(id=scope_id),
        )

    async def upload_file(self, scope_id: str, upload: UploadFile) -> FileMetadata:
        storage = self._require_storage()
        content = await upload.read()
        filename = upload.filename or "upload"
        mime_type = _detect_mime_from_bytes(content)

        prefix = self._uploads_prefix(scope_id)
        await storage.write_file(prefix, filename, content, mime_type)

        file_info = await storage.stat_file(prefix, filename)
        if file_info is None:
            raise HTTPException(
                status_code=500, detail="File written but could not be read back."
            )
        file_info = file_info.model_copy(update={"path": f"uploads/{filename}"})
        return self._to_metadata(file_info, scope_id)

    async def list_files(
        self,
        scope_id: str,
        limit: int = 20,
        after_id: str | None = None,
        before_id: str | None = None,
    ) -> FileListResponse:
        storage = self._require_storage()

        uploads = await storage.list_files_meta(self._uploads_prefix(scope_id))
        outputs = await storage.list_files_meta(self._outputs_prefix(scope_id))

        all_infos = sorted(
            [
                *[
                    fi.model_copy(update={"path": f"uploads/{fi.path}"})
                    for fi in uploads
                ],
                *[
                    fi.model_copy(update={"path": f"outputs/{fi.path}"})
                    for fi in outputs
                ],
            ],
            key=lambda fi: fi.created_at,
        )
        all_files = [self._to_metadata(fi, scope_id) for fi in all_infos]

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

    async def get_file_metadata(self, scope_id: str, file_id: str) -> FileMetadata:
        storage = self._require_storage()
        canonical = _decode_file_id(file_id)
        storage_path = _canonical_to_storage_path(canonical)
        self._validate_file_id(storage_path)
        folder, filename = storage_path.split("/", 1)
        prefix = self._prefix_for_path(storage_path, scope_id)
        file_info = await storage.stat_file(prefix, filename)
        if file_info is None:
            raise HTTPException(status_code=404, detail=f"File '{file_id}' not found.")
        file_info = file_info.model_copy(update={"path": f"{folder}/{filename}"})
        return self._to_metadata(file_info, scope_id)

    async def get_file_content(
        self, scope_id: str, file_id: str
    ) -> tuple[bytes, str, str]:
        """Returns (bytes, mime_type, display_filename)."""
        storage = self._require_storage()
        canonical = _decode_file_id(file_id)
        storage_path = _canonical_to_storage_path(canonical)
        self._validate_file_id(storage_path)
        _folder, filename = storage_path.split("/", 1)
        prefix = self._prefix_for_path(storage_path, scope_id)
        file_info = await storage.stat_file(prefix, filename)
        if file_info is None:
            raise HTTPException(status_code=404, detail=f"File '{file_id}' not found.")
        content = await storage.read_file(prefix, filename)
        display_name = canonical.split("/")[-1]
        return content, file_info.mime_type, display_name

    async def delete_file(self, scope_id: str, file_id: str) -> DeletedFile:
        storage = self._require_storage()
        canonical = _decode_file_id(file_id)
        storage_path = _canonical_to_storage_path(canonical)
        self._validate_file_id(storage_path)
        if not storage_path.startswith("uploads/"):
            raise HTTPException(
                status_code=404,
                detail=f"File '{file_id}' not found or is a sandbox output (cannot be deleted).",
            )
        _folder, filename = storage_path.split("/", 1)
        deleted = await storage.delete_file(self._uploads_prefix(scope_id), filename)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"File '{file_id}' not found or is a sandbox output (cannot be deleted).",
            )
        return DeletedFile(id=file_id)

    @staticmethod
    def _validate_file_id(file_id: str) -> None:
        if ".." in file_id.split("/"):
            raise HTTPException(status_code=400, detail="Invalid file ID.")
