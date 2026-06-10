import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from anyio import to_thread

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


class S3ObjectStorage(ObjectStorage):
    def __init__(self, s3_helper: S3Helper, bucket_name: str) -> None:
        self._s3_helper = s3_helper
        self._bucket_name = bucket_name

    async def write_bundle(self, prefix: str, files: list[StoredFile]) -> None:
        await to_thread.run_sync(self._write_bundle_sync, prefix, files)

    def _write_bundle_sync(self, prefix: str, files: list[StoredFile]) -> None:
        for file in files:
            object_name = f"{prefix}/{file.path}"
            self._s3_helper.upload_file_to_s3(
                filename=file.path,
                bytes_data=file.content,
                bucket_name=self._bucket_name,
                object_name=object_name,
                mime_type=file.mime_type,
            )

    async def delete_prefix(self, prefix: str) -> None:
        await to_thread.run_sync(self._delete_prefix_sync, prefix)

    def _delete_prefix_sync(self, prefix: str) -> None:
        keys = self._s3_helper.list_objects_by_prefix(
            bucket_name=self._bucket_name,
            prefix=prefix,
        )
        for key in keys:
            self._s3_helper.remove_file_from_s3(f"s3://{self._bucket_name}/{key}")

    async def read_file(self, prefix: str, path: str) -> bytes:
        return await to_thread.run_sync(self._read_file_sync, prefix, path)

    def _read_file_sync(self, prefix: str, path: str) -> bytes:
        s3_url = f"s3://{self._bucket_name}/{prefix}/{path}"
        binary = self._s3_helper.load_file_from_s3(s3_url)
        return binary.read()

    async def list_files(self, prefix: str) -> list[str]:
        return await to_thread.run_sync(self._list_files_sync, prefix)

    def _list_files_sync(self, prefix: str) -> list[str]:
        keys = self._s3_helper.list_objects_by_prefix(
            bucket_name=self._bucket_name,
            prefix=prefix,
        )
        return [k[len(prefix) :].lstrip("/") for k in keys]
