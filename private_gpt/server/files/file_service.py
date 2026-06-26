from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import magic
from fastapi import HTTPException
from injector import inject, singleton

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

    def _prefix(self, scope_id: str) -> str:
        return f"{self._settings.code_execution.vfs_sessions_prefix}/{scope_id}"

    def _to_metadata(self, file_info: FileInfo, scope_id: str) -> FileMetadata:
        downloadable = not file_info.path.startswith("uploads/")
        return FileMetadata(
            id=file_info.path,
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
        path = f"uploads/{filename}"
        mime_type = _detect_mime_from_bytes(content)

        await storage.write_file(self._prefix(scope_id), path, content, mime_type)

        file_info = await storage.stat_file(self._prefix(scope_id), path)
        if file_info is None:
            raise HTTPException(
                status_code=500, detail="File written but could not be read back."
            )
        return self._to_metadata(file_info, scope_id)

    async def list_files(
        self,
        scope_id: str,
        limit: int = 20,
        after_id: str | None = None,
        before_id: str | None = None,
    ) -> FileListResponse:
        storage = self._require_storage()
        prefix = self._prefix(scope_id)

        uploads = await storage.list_files_meta(f"{prefix}/uploads")
        outputs = await storage.list_files_meta(f"{prefix}/outputs")

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
        self._validate_file_id(file_id)
        file_info = await storage.stat_file(self._prefix(scope_id), file_id)
        if file_info is None:
            raise HTTPException(status_code=404, detail=f"File '{file_id}' not found.")
        return self._to_metadata(file_info, scope_id)

    async def get_file_content(
        self, scope_id: str, file_id: str
    ) -> tuple[bytes, str, str]:
        """Returns (bytes, mime_type, display_filename)."""
        storage = self._require_storage()
        self._validate_file_id(file_id)
        file_info = await storage.stat_file(self._prefix(scope_id), file_id)
        if file_info is None:
            raise HTTPException(status_code=404, detail=f"File '{file_id}' not found.")
        content = await storage.read_file(self._prefix(scope_id), file_id)
        filename = file_id.split("/")[-1]
        return content, file_info.mime_type, filename

    async def delete_file(self, scope_id: str, file_id: str) -> DeletedFile:
        storage = self._require_storage()
        self._validate_file_id(file_id)
        if not file_id.startswith("uploads/"):
            raise HTTPException(
                status_code=404,
                detail=f"File '{file_id}' not found or is a sandbox output (cannot be deleted).",
            )
        deleted = await storage.delete_file(self._prefix(scope_id), file_id)
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
