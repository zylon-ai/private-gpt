from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import magic
from anyio import to_thread

from private_gpt.components.storage.models import FileInfo

if TYPE_CHECKING:
    from private_gpt.components.storage.models import StoredFile
    from private_gpt.components.storage.s3_helper import S3Helper


class ObjectStorage(ABC):
    @abstractmethod
    async def write_bundle(self, prefix: str, files: list[StoredFile]) -> None:
        ...

    @abstractmethod
    async def delete_prefix(self, prefix: str) -> None:
        ...

    @abstractmethod
    async def read_file(self, prefix: str, path: str) -> bytes:
        ...

    @abstractmethod
    async def list_files(self, prefix: str) -> list[str]:
        """List all file paths relative to prefix."""
        ...

    @abstractmethod
    async def write_file(
        self,
        prefix: str,
        path: str,
        content: bytes,
        mime_type: str = "application/octet-stream",
    ) -> None:
        """Write a single file at prefix/path. Creates parent directories as needed."""
        ...

    @abstractmethod
    async def stat_file(self, prefix: str, path: str) -> FileInfo | None:
        """Return metadata for a single file, or None if it does not exist."""
        ...

    @abstractmethod
    async def list_files_meta(self, prefix: str) -> list[FileInfo]:
        """List files under prefix with metadata (non-recursive, top-level only)."""
        ...

    @abstractmethod
    async def delete_file(self, prefix: str, path: str) -> bool:
        """Delete a single file. Returns True if the file existed and was deleted."""
        ...


class LocalObjectStorage(ObjectStorage):
    def __init__(self, root_path: str) -> None:
        self._root_path = root_path

    async def write_bundle(self, prefix: str, files: list[StoredFile]) -> None:
        await to_thread.run_sync(self._write_bundle_sync, prefix, files)

    def _write_bundle_sync(self, prefix: str, files: list[StoredFile]) -> None:
        for file in files:
            destination = Path(self._root_path) / prefix / file.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(file.content)

    async def delete_prefix(self, prefix: str) -> None:
        await to_thread.run_sync(self._delete_prefix_sync, prefix)

    def _delete_prefix_sync(self, prefix: str) -> None:
        path = Path(self._root_path) / prefix
        if path.exists():
            shutil.rmtree(path)

    async def read_file(self, prefix: str, path: str) -> bytes:
        return await to_thread.run_sync(self._read_file_sync, prefix, path)

    def _read_file_sync(self, prefix: str, path: str) -> bytes:
        target = Path(self._root_path) / prefix / path
        return target.read_bytes()

    async def list_files(self, prefix: str) -> list[str]:
        return await to_thread.run_sync(self._list_files_sync, prefix)

    def _list_files_sync(self, prefix: str) -> list[str]:
        root = Path(self._root_path) / prefix
        if not root.exists():
            return []
        return [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()]

    async def write_file(
        self,
        prefix: str,
        path: str,
        content: bytes,
        mime_type: str = "application/octet-stream",
    ) -> None:
        await to_thread.run_sync(self._write_file_sync, prefix, path, content)

    def _write_file_sync(self, prefix: str, path: str, content: bytes) -> None:
        dest = Path(self._root_path) / prefix / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)

    async def stat_file(self, prefix: str, path: str) -> FileInfo | None:
        return await to_thread.run_sync(self._stat_file_sync, prefix, path)

    def _stat_file_sync(self, prefix: str, path: str) -> FileInfo | None:
        target = Path(self._root_path) / prefix / path
        if not target.exists() or not target.is_file():
            return None
        stat = target.stat()
        try:
            mime = magic.Magic(mime=True).from_file(str(target))
        except Exception:
            mime = "application/octet-stream"
        return FileInfo(
            path=path,
            size_bytes=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            mime_type=mime,
        )

    async def list_files_meta(self, prefix: str) -> list[FileInfo]:
        return await to_thread.run_sync(self._list_files_meta_sync, prefix)

    def _list_files_meta_sync(self, prefix: str) -> list[FileInfo]:
        root = Path(self._root_path) / prefix
        if not root.exists():
            return []
        results: list[FileInfo] = []
        for entry in root.iterdir():
            if not entry.is_file():
                continue
            stat = entry.stat()
            try:
                mime = magic.Magic(mime=True).from_file(str(entry))
            except Exception:
                mime = "application/octet-stream"
            results.append(
                FileInfo(
                    path=str(entry.relative_to(root)),
                    size_bytes=stat.st_size,
                    created_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                    mime_type=mime,
                )
            )
        return results

    async def delete_file(self, prefix: str, path: str) -> bool:
        return await to_thread.run_sync(self._delete_file_sync, prefix, path)

    def _delete_file_sync(self, prefix: str, path: str) -> bool:
        target = Path(self._root_path) / prefix / path
        if not target.exists():
            return False
        target.unlink()
        return True


class S3ObjectStorage(ObjectStorage):
    def __init__(self, s3_helper: S3Helper, bucket_name: str) -> None:
        self._s3_helper = s3_helper
        self._bucket_name = bucket_name

    async def write_bundle(self, prefix: str, files: list[StoredFile]) -> None:
        for file in files:
            await self._s3_helper.async_upload_file_to_s3(
                filename=file.path,
                bytes_data=file.content,
                bucket_name=self._bucket_name,
                object_name=f"{prefix}/{file.path}",
                mime_type=file.mime_type,
            )

    async def delete_prefix(self, prefix: str) -> None:
        keys = await self._s3_helper.async_list_objects_by_prefix(
            bucket_name=self._bucket_name,
            prefix=prefix,
        )
        for key in keys:
            await self._s3_helper.async_remove_file_from_s3(
                f"s3://{self._bucket_name}/{key}"
            )

    async def read_file(self, prefix: str, path: str) -> bytes:
        binary = await self._s3_helper.async_load_file_from_s3(
            f"s3://{self._bucket_name}/{prefix}/{path}"
        )
        return binary.read()

    async def list_files(self, prefix: str) -> list[str]:
        keys = await self._s3_helper.async_list_objects_by_prefix(
            bucket_name=self._bucket_name,
            prefix=prefix,
        )
        return [key[len(prefix) :].lstrip("/") for key in keys]

    async def write_file(
        self,
        prefix: str,
        path: str,
        content: bytes,
        mime_type: str = "application/octet-stream",
    ) -> None:
        await self._s3_helper.async_upload_file_to_s3(
            filename=path,
            bytes_data=content,
            bucket_name=self._bucket_name,
            object_name=f"{prefix}/{path}",
            mime_type=mime_type,
        )

    async def stat_file(self, prefix: str, path: str) -> FileInfo | None:
        meta = await self._s3_helper.async_head_object(
            self._bucket_name, f"{prefix}/{path}"
        )
        return self._file_info(path, meta)

    async def list_files_meta(self, prefix: str) -> list[FileInfo]:
        keys = await self._s3_helper.async_list_objects_by_prefix(
            bucket_name=self._bucket_name,
            prefix=prefix,
        )
        results: list[FileInfo] = []
        for key in keys:
            relative_path = key[len(prefix) :].lstrip("/")
            if not relative_path:
                continue
            meta = await self._s3_helper.async_head_object(self._bucket_name, key)
            file_info = self._file_info(relative_path, meta)
            if file_info is not None:
                results.append(file_info)
        return results

    async def delete_file(self, prefix: str, path: str) -> bool:
        return await self._s3_helper.async_delete_key(
            self._bucket_name, f"{prefix}/{path}"
        )

    @staticmethod
    def _file_info(path: str, meta: dict[str, object] | None) -> FileInfo | None:
        if meta is None:
            return None
        last_modified = meta.get("last_modified")
        created_at = (
            last_modified
            if isinstance(last_modified, datetime)
            else datetime.now(tz=UTC)
        )
        return FileInfo(
            path=path,
            size_bytes=int(str(meta.get("content_length", 0))),
            created_at=created_at,
            mime_type=str(meta.get("content_type", "application/octet-stream")),
        )
