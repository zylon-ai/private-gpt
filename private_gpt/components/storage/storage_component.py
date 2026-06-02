import logging
import threading

from injector import Injector, inject, singleton

from private_gpt.components.storage.object_storage import (
    LocalObjectStorage,
    ObjectStorage,
    S3ObjectStorage,
)
from private_gpt.components.storage.s3_helper import S3Helper

logger = logging.getLogger(__name__)


@singleton
class StorageComponent:
    @inject
    def __init__(self, injector: Injector) -> None:
        self._injector = injector
        self._lock = threading.RLock()
        self._storages: dict[str, ObjectStorage] = {}

    def get_object_storage(
        self,
        provider: str,
        local_root_path: str | None = None,
        bucket_name: str | None = None,
    ) -> ObjectStorage:
        key = f"{provider}:{local_root_path}:{bucket_name}"
        with self._lock:
            storage = self._storages.get(key)
            if storage is not None:
                return storage

            match provider:
                case "local":
                    if local_root_path is None:
                        raise ValueError(
                            "Local storage provider requires local_root_path"
                        )

                    storage = LocalObjectStorage(root_path=local_root_path)
                case "s3":
                    if bucket_name is None:
                        raise ValueError("S3 storage provider requires bucket_name")
                    storage = S3ObjectStorage(
                        s3_helper=self._injector.get(S3Helper),
                        bucket_name=bucket_name,
                    )
                case _:
                    raise ValueError(f"Unsupported storage provider: {provider}")

            self._storages[key] = storage
            return storage
