from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from injector import Injector, inject, singleton

from private_gpt.components.ingest.utils import get_extension, get_file_name
from private_gpt.components.storage.s3_helper import S3Helper
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.settings.settings import Settings, settings

if TYPE_CHECKING:
    from private_gpt.server.ingest.ingest_router import (
        DeleteIngestedDocumentAsyncBody,
        IngestAsyncBody,
        IngestBody,
        IngestResponse,
    )

logger = logging.getLogger(__name__)

IngestionSchedulerProvider = (
    type["BaseIngestionScheduler"] | Callable[[Injector], "BaseIngestionScheduler"]
)
_INGESTION_SCHEDULERS: dict[str, IngestionSchedulerProvider] = {}


def register_ingestion_scheduler(
    mode: str, provider: IngestionSchedulerProvider
) -> None:
    _INGESTION_SCHEDULERS[mode] = provider


class BaseIngestionScheduler(ABC):
    """Shared ingestion logic between local and Celery modes."""

    @property
    def is_async(self) -> bool:
        return False

    def __init__(
        self,
        ingest_service: IngestService,
        s3_helper: S3Helper | None = None,
    ) -> None:
        self._ingest_service = ingest_service
        self._s3_helper = s3_helper

    def _require_s3_helper(self) -> S3Helper:
        if self._s3_helper is None:
            raise RuntimeError("S3Helper is required for remote ingestion flows")
        return self._s3_helper

    @abstractmethod
    def ingest(self, ingest_body: IngestBody) -> IngestResponse: ...

    def ingest_async(self, ingest_body: IngestAsyncBody) -> str:
        del ingest_body
        raise NotImplementedError

    def delete(self, collection: str, artifact: str) -> None:
        self._ingest_service.delete(collection=collection, artifact=artifact)

    def delete_async(self, delete_body: DeleteIngestedDocumentAsyncBody) -> str:
        del delete_body
        raise NotImplementedError


@singleton
class LocalIngestionScheduler(BaseIngestionScheduler):
    """Run sync ingestion in-process."""

    @inject
    def __init__(self, ingest_service: IngestService) -> None:
        super().__init__(ingest_service)

    def ingest(self, ingest_body: IngestBody) -> IngestResponse:
        from private_gpt.server.ingest.ingest_router import ingest_data_sync

        content = ingest_body.input.to_binary_content(
            get_file_name(ingest_body.metadata)
        )
        with self._ingest_service.temporary_file(
            lambda: self._ingest_service.data_path_from_bin_data(
                content.data, get_extension(content.filename)
            )
        ) as data_path:
            return ingest_data_sync(
                collection=ingest_body.collection,
                artifact=ingest_body.artifact,
                data_path=data_path,
                service=self._ingest_service,
                metadata={
                    **(ingest_body.metadata or {}),
                    "file_name": content.filename,
                },
            )


@singleton
class CeleryIngestionScheduler(BaseIngestionScheduler):
    """Dispatch sync ingestion to a Celery worker and wait."""

    @inject
    def __init__(
        self, settings: Settings, s3_helper: S3Helper, ingest_service: IngestService
    ) -> None:
        super().__init__(ingest_service, s3_helper)
        self._settings = settings

    @property
    def is_async(self) -> bool:
        return True

    def ingest_async(self, ingest_body: IngestAsyncBody) -> str:
        import uuid

        from private_gpt.celery.dispatch import dispatch_task
        from private_gpt.celery.tasks.ingestion.extraction_tasks import (
            VECTOR_INDEX_TASK_NAME,
        )
        from private_gpt.server.utils.artifact_input import UriArtifact

        config = settings()
        self._ingest_service.initialize_artifact_indices(
            collection=ingest_body.ingest_body.collection,
            artifact=ingest_body.ingest_body.artifact,
        )
        if ingest_body.ingest_body.input:
            should_upload = (
                not isinstance(ingest_body.ingest_body.input, UriArtifact)
                or ingest_body.ingest_body.input.is_base64()
            )
            s3_helper = self._require_s3_helper()
            if should_upload and s3_helper.is_available():
                content = ingest_body.ingest_body.input.to_binary_content()
                object_name = str(uuid.uuid4())
                s3_url = s3_helper.upload_file_to_s3(
                    filename=content.filename,
                    bytes_data=content.data.read(),
                    bucket_name=config.s3.temporary_bucket_name,
                    object_name=object_name,
                )
                if not s3_url:
                    raise ValueError("Failed to upload file to S3 for async ingestion")
                ingest_body.ingest_body.input = UriArtifact(value=s3_url)

        result = dispatch_task(
            task_name=VECTOR_INDEX_TASK_NAME,
            args=(ingest_body,),
            queue=config.scheduler.ingestion.celery_queue,
        )
        task_id = result.task_id
        if not isinstance(task_id, str):
            raise ValueError("Async ingestion task did not return a valid task_id")
        return task_id

    def delete_async(self, delete_body: DeleteIngestedDocumentAsyncBody) -> str:
        from private_gpt.celery.celery import celery_app
        from private_gpt.celery.dispatch import dispatch_task
        from private_gpt.celery.task_helper import IngestionTaskHelper
        from private_gpt.celery.tasks.ingestion.delete_tasks import (
            DELETE_INGESTED_TASK_NAME,
        )

        revoked = IngestionTaskHelper.revoke_ingestion_task(
            celery_app=celery_app,
            collection=delete_body.delete_body.collection,
            artifact=delete_body.delete_body.artifact,
        )
        if revoked:
            return "revoked"

        config = settings()
        result = dispatch_task(
            task_name=DELETE_INGESTED_TASK_NAME,
            args=(delete_body,),
            queue=config.scheduler.ingestion.celery_queue,
        )
        task_id = result.task_id
        if not isinstance(task_id, str):
            raise ValueError("Delete ingestion task did not return a valid task_id")
        return task_id

    def ingest(self, ingest_body: IngestBody) -> IngestResponse:
        import uuid

        from private_gpt.celery.dispatch import dispatch_task
        from private_gpt.celery.tasks.ingestion.extraction_tasks import (
            VECTOR_INDEX_TASK_NAME,
        )
        from private_gpt.server.ingest.ingest_router import (
            IngestAsyncBody,
            IngestResponse,
        )
        from private_gpt.server.utils.artifact_input import UriArtifact
        from private_gpt.server.utils.callback import AMQP, Callback

        config = settings()
        self._ingest_service.initialize_artifact_indices(
            collection=ingest_body.collection,
            artifact=ingest_body.artifact,
        )
        async_body = IngestAsyncBody(
            ingest_body=ingest_body,
            callback=Callback(
                amqp=AMQP(
                    exchange="ingestion_events",
                    routing_key_done="ingest.completed",
                    routing_key_error="ingest.failed",
                    routing_key_progress="ingest.progress",
                )
            ),
        )

        if async_body.ingest_body.input:
            s3_helper = self._require_s3_helper()
            if s3_helper.is_available():
                content = async_body.ingest_body.input.to_binary_content(
                    async_body.ingest_body.metadata.get("file_name")
                    if async_body.ingest_body.metadata
                    else None
                )
                object_name = str(uuid.uuid4())
                s3_url = s3_helper.upload_file_to_s3(
                    filename=content.filename,
                    bytes_data=content.data.read(),
                    bucket_name=config.s3.temporary_bucket_name,
                    object_name=object_name,
                )
                if not s3_url:
                    raise ValueError(
                        "Failed to upload file to S3 for sync celery ingest"
                    )
                async_body.ingest_body.input = UriArtifact(value=s3_url)

        result = dispatch_task(
            task_name=VECTOR_INDEX_TASK_NAME,
            args=(async_body,),
            queue=config.scheduler.ingestion.celery_queue,
        )
        while not result.ready():
            import time

            time.sleep(0.1)
        if result.failed():
            raise result.result
        return IngestResponse.model_validate(result.result)


register_ingestion_scheduler("local", LocalIngestionScheduler)
register_ingestion_scheduler("celery", CeleryIngestionScheduler)


@singleton
class IngestionSchedulerFactory:
    @inject
    def __init__(self, settings: Settings, injector: Injector) -> None:
        self._settings = settings
        self._injector = injector
        self._scheduler: BaseIngestionScheduler | None = None

    def get(self) -> BaseIngestionScheduler:
        if self._scheduler is None:
            mode = self._settings.scheduler.ingestion.mode
            provider = _INGESTION_SCHEDULERS.get(mode)
            if provider is None:
                raise ValueError(f"Unknown scheduler.ingestion.mode: {mode}")
            self._scheduler = (
                self._injector.get(provider)
                if isinstance(provider, type)
                else provider(self._injector)
            )
        return self._scheduler
