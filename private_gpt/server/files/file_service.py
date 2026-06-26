from datetime import UTC, datetime
from pathlib import Path

import anyio
import magic
from fastapi import HTTPException, UploadFile
from injector import inject, singleton

from private_gpt.components.code_execution.code_execution_component import (
    CodeExecutionComponent,
)
from private_gpt.components.environment.session_volume_locator import (
    SessionVolumeLocator,
)
from private_gpt.server.files.file_models import (
    DeletedFile,
    FileListResponse,
    FileMetadata,
    FileScope,
)


def _detect_mime(path: Path) -> str:
    try:
        return magic.Magic(mime=True).from_file(str(path))
    except Exception:
        return "application/octet-stream"


@singleton
class FileService:
    @inject
    def __init__(
        self,
        locator: SessionVolumeLocator,
        code_execution: CodeExecutionComponent,
    ) -> None:
        self._locator = locator
        self._code_execution = code_execution

    async def _ensure_session(self, scope_id: str) -> None:
        await self._code_execution.get_or_create_session(scope_id)

    def _session_root(self, scope_id: str) -> Path:
        return self._locator.uploads_path(scope_id).parent

    def _is_under(self, path: Path, base: Path) -> bool:
        try:
            path.resolve().relative_to(base.resolve())
            return True
        except ValueError:
            return False

    def _make_entry(
        self, path: Path, scope_id: str, downloadable: bool
    ) -> FileMetadata:
        stat = path.stat()
        return FileMetadata(
            id=str(path.resolve()),
            created_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            filename=path.name,
            mime_type=_detect_mime(path),
            size_bytes=stat.st_size,
            downloadable=downloadable,
            scope=FileScope(id=scope_id),
        )

    def _load_upload_entries(self, scope_id: str) -> list[FileMetadata]:
        uploads_dir = self._locator.uploads_path(scope_id)
        if not uploads_dir.exists():
            return []
        return [
            self._make_entry(p, scope_id, downloadable=False)
            for p in uploads_dir.iterdir()
            if p.is_file()
        ]

    def _load_output_entries(self, scope_id: str) -> list[FileMetadata]:
        outputs_dir = self._locator.outputs_path(scope_id)
        if not outputs_dir.exists():
            return []
        return [
            self._make_entry(p, scope_id, downloadable=True)
            for p in outputs_dir.iterdir()
            if p.is_file()
        ]

    async def upload_file(self, scope_id: str, upload: UploadFile) -> FileMetadata:
        await self._ensure_session(scope_id)
        bytes_data = await upload.read()
        filename = upload.filename or "upload"
        dest = self._locator.uploads_path(scope_id) / filename

        def _write() -> None:
            dest.write_bytes(bytes_data)

        await anyio.to_thread.run_sync(_write)

        def _make() -> FileMetadata:
            return self._make_entry(dest, scope_id, downloadable=False)

        return await anyio.to_thread.run_sync(_make)

    async def list_files(
        self,
        scope_id: str,
        limit: int = 20,
        after_id: str | None = None,
        before_id: str | None = None,
    ) -> FileListResponse:
        def _collect() -> list[FileMetadata]:
            return sorted(
                self._load_upload_entries(scope_id)
                + self._load_output_entries(scope_id),
                key=lambda f: f.created_at,
            )

        all_files = await anyio.to_thread.run_sync(_collect)

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
        def _load() -> FileMetadata | None:
            path = Path(file_id)
            if not path.exists() or not path.is_file():
                return None
            if not self._is_under(path, self._session_root(scope_id)):
                return None
            downloadable = not self._is_under(
                path, self._locator.uploads_path(scope_id)
            )
            return self._make_entry(path, scope_id, downloadable=downloadable)

        result = await anyio.to_thread.run_sync(_load)
        if result is None:
            raise HTTPException(status_code=404, detail=f"File '{file_id}' not found.")
        return result

    async def get_file_content(
        self, scope_id: str, file_id: str
    ) -> tuple[bytes, str, str]:
        """Returns (bytes, mime_type, display_filename)."""

        def _read() -> tuple[bytes, str, str] | None:
            path = Path(file_id)
            if not path.exists() or not path.is_file():
                return None
            if not self._is_under(path, self._session_root(scope_id)):
                return None
            return path.read_bytes(), _detect_mime(path), path.name

        result = await anyio.to_thread.run_sync(_read)
        if result is None:
            raise HTTPException(status_code=404, detail=f"File '{file_id}' not found.")
        return result

    async def delete_file(self, scope_id: str, file_id: str) -> DeletedFile:
        def _delete() -> bool:
            path = Path(file_id)
            if not self._is_under(path, self._locator.uploads_path(scope_id)):
                return False
            if not path.exists():
                return False
            path.unlink()
            return True

        deleted = await anyio.to_thread.run_sync(_delete)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"File '{file_id}' not found or is a sandbox output (cannot be deleted).",
            )
        return DeletedFile(id=file_id)
