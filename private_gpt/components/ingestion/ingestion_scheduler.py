from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from injector import Injector, inject, singleton

from private_gpt.celery.result import wait_for_celery_result
from private_gpt.components.ingest.utils import get_extension, get_file_name
from private_gpt.components.storage.s3_helper import S3Helper
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.settings.settings import Settings, settings

if TYPE_CHECKING:
    from llama_index.core.schema import BaseNode

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
    """Normalized ingestion scheduler interface.

    Every operation comes in two flavours:
    - sync  : runs to completion and returns the result.
    - async : fire-and-forget; dispatches work and returns a task-id string.

    Operations
    ----------
    parse         - validate + read a file into list[BaseNode] (no vectorisation).
    store         - embed + persist a list of BaseNode objects.
    ingest        - parse + store combined (convenience for the sync HTTP endpoint).
    delete        - remove an artifact from all indexes.
    bytes_to_text - parse raw bytes to plain text (satisfies DocumentConverter).
    """

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

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    @abstractmethod
    def parse(self, ingest_body: IngestBody) -> list[BaseNode]:
        """Parse a file into tree nodes in-process (sync)."""
        ...

    def parse_async(self, ingest_body: IngestAsyncBody) -> str:
        """Dispatch parse-only work and return the task-id (fire-and-forget)."""
        del ingest_body
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    @abstractmethod
    def store(self, nodes: list[BaseNode], ingest_body: IngestBody) -> IngestResponse:
        """Embed and persist pre-parsed nodes in-process (sync)."""
        ...

    def store_async(self, ingest_body: IngestAsyncBody) -> str:
        """Dispatch store-only work (nodes already on the body) and return task-id."""
        del ingest_body
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Ingest  (parse + store combined)
    # ------------------------------------------------------------------

    @abstractmethod
    def ingest(self, ingest_body: IngestBody) -> IngestResponse:
        """Parse and store in-process (sync)."""
        ...

    async def ingest_for_request(self, ingest_body: IngestBody) -> IngestResponse:
        return await asyncio.to_thread(self.ingest, ingest_body)

    def ingest_async(self, ingest_body: IngestAsyncBody) -> str:
        """Dispatch parse→store pipeline, return parse task-id (fire-and-forget)."""
        del ingest_body
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, collection: str, artifact: str) -> None:
        self._ingest_service.delete(collection=collection, artifact=artifact)

    def delete_async(self, delete_body: DeleteIngestedDocumentAsyncBody) -> str:
        del delete_body
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convert (DocumentConverter protocol)
    # ------------------------------------------------------------------

    def bytes_to_text(self, raw: bytes, ext: str) -> str:
        """Parse raw bytes to plain text via the configured execution path.

        Satisfies the ``DocumentConverter`` protocol so chat document
        preprocessing routes through the scheduler instead of calling
        ``ParseComponent`` directly.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Local (in-process) implementation
# ---------------------------------------------------------------------------


@singleton
class LocalIngestionScheduler(BaseIngestionScheduler):
    """Run all operations in-process."""

    @inject
    def __init__(self, ingest_service: IngestService) -> None:
        super().__init__(ingest_service)

    def parse(self, ingest_body: IngestBody) -> list[BaseNode]:
        from private_gpt.components.ingest.utils import get_extension, get_file_name

        content = ingest_body.input.to_binary_content(
            get_file_name(ingest_body.metadata)
        )
        with self._ingest_service.temporary_file(
            lambda: self._ingest_service.data_path_from_bin_data(
                content.data, get_extension(content.filename)
            )
        ) as data_path:
            self._ingest_service.initialize_artifact_indices(
                collection=ingest_body.collection,
                artifact=ingest_body.artifact,
            )
            file_info, _, warnings = (
                self._ingest_service.parse_component.load_and_validate_file(
                    file_data=data_path,
                    file_metadata={
                        **(ingest_body.metadata or {}),
                        "file_name": content.filename,
                    },
                )
            )
            return self._ingest_service.ingest_component.parse_file_into_nodes(
                artifact=ingest_body.artifact,
                collection=ingest_body.collection,
                file_info=file_info,
                file_metadata={
                    **(ingest_body.metadata or {}),
                    "file_name": content.filename,
                },
                warnings=warnings,
            )

    def store(self, nodes: list[BaseNode], ingest_body: IngestBody) -> IngestResponse:
        from private_gpt.server.ingest.ingest_router import IngestResponse
        from private_gpt.server.ingest.model import IngestedDoc

        self._ingest_service.initialize_artifact_indices(
            collection=ingest_body.collection,
            artifact=ingest_body.artifact,
        )
        vector_artifact_index = self._ingest_service._make_vector_artifact_index(
            collection=ingest_body.collection,
            artifact=ingest_body.artifact,
        )
        from llama_index.core import StorageContext, load_index_from_storage

        index = load_index_from_storage(
            index_id=vector_artifact_index.index_id(),
            storage_context=StorageContext.from_defaults(
                vector_store=self._ingest_service.vector_store_component.vector_store(
                    ingest_body.collection
                ),
                index_store=self._ingest_service.node_store_component.index_store(
                    ingest_body.collection
                ),
            ),
            embed_model=self._ingest_service.embedding_component.get_embed(),
            transformations=[],
            show_progress=False,
            use_async=False,
            insert_batch_size=512,
        )
        self._ingest_service.ingest_component.load_index(
            artifact=ingest_body.artifact,
            collection=ingest_body.collection,
            index=index,
            index_id=vector_artifact_index.index_id(),
            nodes=nodes,
        )
        return IngestResponse(
            object="list",
            model="private-gpt",
            data=[IngestedDoc.from_document(nodes[0])] if nodes else [],
        )

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

    def bytes_to_text(self, raw: bytes, ext: str) -> str:
        from private_gpt.server.ingest.convert_service import ConvertService

        return ConvertService(self._ingest_service.parse_component).bytes_to_text(
            raw, ext
        )


# ---------------------------------------------------------------------------
# Celery implementation
# ---------------------------------------------------------------------------


@singleton
class CeleryIngestionScheduler(BaseIngestionScheduler):
    """Dispatch work to Celery workers."""

    @inject
    def __init__(
        self, settings: Settings, s3_helper: S3Helper, ingest_service: IngestService
    ) -> None:
        super().__init__(ingest_service, s3_helper)
        self._settings = settings

    @property
    def is_async(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    def parse(self, ingest_body: IngestBody) -> list[BaseNode]:
        """Parse in-process on the calling side (no worker dispatch).

        For Celery mode the parse step runs on a worker via ``parse_async``.
        This sync variant runs in-process and is provided for completeness /
        testing; prefer ``parse_async`` in production.
        """
        from private_gpt.components.ingest.utils import get_extension, get_file_name

        content = ingest_body.input.to_binary_content(
            get_file_name(ingest_body.metadata)
        )
        with self._ingest_service.temporary_file(
            lambda: self._ingest_service.data_path_from_bin_data(
                content.data, get_extension(content.filename)
            )
        ) as data_path:
            self._ingest_service.initialize_artifact_indices(
                collection=ingest_body.collection,
                artifact=ingest_body.artifact,
            )
            file_info, _, warnings = (
                self._ingest_service.parse_component.load_and_validate_file(
                    file_data=data_path,
                    file_metadata={
                        **(ingest_body.metadata or {}),
                        "file_name": content.filename,
                    },
                )
            )
            return self._ingest_service.ingest_component.parse_file_into_nodes(
                artifact=ingest_body.artifact,
                collection=ingest_body.collection,
                file_info=file_info,
                file_metadata={
                    **(ingest_body.metadata or {}),
                    "file_name": content.filename,
                },
                warnings=warnings,
            )

    def parse_async(self, ingest_body: IngestAsyncBody) -> str:
        """Dispatch parse_task (no store chain) and return its task-id."""
        from private_gpt.celery.dispatch import dispatch_task
        from private_gpt.celery.tasks.ingestion.extraction_tasks import PARSE_TASK_NAME

        config = settings()
        result = dispatch_task(
            task_name=PARSE_TASK_NAME,
            args=(ingest_body,),
            kwargs={"dispatch_store": False},
            queue=config.scheduler.ingestion.celery_queue,
        )
        task_id = result.task_id
        if not isinstance(task_id, str):
            raise ValueError("parse_async did not return a valid task_id")
        return task_id

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(self, nodes: list[BaseNode], ingest_body: IngestBody) -> IngestResponse:
        """Store in-process on the calling side (no worker dispatch)."""
        from private_gpt.server.ingest.ingest_router import IngestResponse
        from private_gpt.server.ingest.model import IngestedDoc

        vector_artifact_index = self._ingest_service._make_vector_artifact_index(
            collection=ingest_body.collection,
            artifact=ingest_body.artifact,
        )
        from llama_index.core import StorageContext, load_index_from_storage

        index = load_index_from_storage(
            index_id=vector_artifact_index.index_id(),
            storage_context=StorageContext.from_defaults(
                vector_store=self._ingest_service.vector_store_component.vector_store(
                    ingest_body.collection
                ),
                index_store=self._ingest_service.node_store_component.index_store(
                    ingest_body.collection
                ),
            ),
            embed_model=self._ingest_service.embedding_component.get_embed(),
            transformations=[],
            show_progress=False,
            use_async=False,
            insert_batch_size=512,
        )
        self._ingest_service.ingest_component.load_index(
            artifact=ingest_body.artifact,
            collection=ingest_body.collection,
            index=index,
            index_id=vector_artifact_index.index_id(),
            nodes=nodes,
        )
        return IngestResponse(
            object="list",
            model="private-gpt",
            data=[IngestedDoc.from_document(nodes[0])] if nodes else [],
        )

    def store_async(self, ingest_body: IngestAsyncBody) -> str:
        """Dispatch store_vectors_task and return its task-id (fire-and-forget).

        ``ingest_body.nodes`` must be populated before calling this method.
        """
        from private_gpt.celery.dispatch import dispatch_task
        from private_gpt.celery.tasks.ingestion.extraction_tasks import (
            STORE_VECTORS_TASK_NAME,
        )

        config = settings()
        result = dispatch_task(
            task_name=STORE_VECTORS_TASK_NAME,
            args=(ingest_body,),
            queue=config.scheduler.ingestion.celery_queue,
        )
        task_id = result.task_id
        if not isinstance(task_id, str):
            raise ValueError("store_async did not return a valid task_id")
        return task_id

    # ------------------------------------------------------------------
    # Ingest  (parse → store pipeline)
    # ------------------------------------------------------------------

    def ingest_async(self, ingest_body: IngestAsyncBody) -> str:
        """Dispatch parse_task chained to store_vectors_task; return parse task-id.

        Fire-and-forget. The AMQP done/error callback fires from
        store_vectors_task under the legacy ``pgpt.vector_index_task.*`` names.
        """
        import uuid

        from private_gpt.celery.dispatch import dispatch_task
        from private_gpt.celery.tasks.ingestion.extraction_tasks import PARSE_TASK_NAME
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
                content = ingest_body.ingest_body.input.to_binary_content(
                    get_file_name(ingest_body.ingest_body.metadata)
                )
                object_name = str(uuid.uuid4())
                s3_url = s3_helper.upload_file_to_s3(
                    filename=content.filename,
                    bytes_data=content.data.read(),
                    bucket_name=config.s3.temporary_bucket_name,
                    object_name=object_name,
                )
                if not s3_url:
                    raise ValueError("Failed to upload file to S3 for async ingestion")
                ingest_body.ingest_body.metadata = {
                    **(ingest_body.ingest_body.metadata or {}),
                    "file_name": content.filename,
                }
                ingest_body.ingest_body.input = UriArtifact(value=s3_url)

        result = dispatch_task(
            task_name=PARSE_TASK_NAME,
            args=(ingest_body,),
            queue=config.scheduler.ingestion.celery_queue,
        )
        task_id = result.task_id
        if not isinstance(task_id, str):
            raise ValueError("ingest_async did not return a valid task_id")
        return task_id

    def ingest(self, ingest_body: IngestBody) -> IngestResponse:
        """Parse then store synchronously, blocking until both complete."""
        import uuid

        from private_gpt.server.ingest.ingest_router import (
            IngestAsyncBody,
            IngestResponse,
        )
        from private_gpt.server.utils.artifact_input import UriArtifact

        config = settings()
        self._ingest_service.initialize_artifact_indices(
            collection=ingest_body.collection,
            artifact=ingest_body.artifact,
        )
        async_body = IngestAsyncBody(ingest_body=ingest_body)

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
                    raise ValueError("Failed to upload file to S3 for sync ingest")
                async_body.ingest_body.metadata = {
                    **(async_body.ingest_body.metadata or {}),
                    "file_name": content.filename,
                }
                async_body.ingest_body.input = UriArtifact(value=s3_url)

        # 1. Parse on worker, wait — fast step.
        from private_gpt.celery.dispatch import dispatch_task
        from private_gpt.celery.tasks.ingestion.extraction_tasks import PARSE_TASK_NAME

        parse_result = dispatch_task(
            task_name=PARSE_TASK_NAME,
            args=(async_body,),
            queue=config.scheduler.ingestion.celery_queue,
        )
        parse_result_value = wait_for_celery_result(parse_result)

        # 2. parse_task returns the store_vectors task_id; poll it.
        assert isinstance(parse_result_value, str)
        from celery.result import AsyncResult

        from private_gpt.celery.celery import celery_app

        store_result = AsyncResult(parse_result_value, app=celery_app)
        return IngestResponse.model_validate(wait_for_celery_result(store_result))

    async def ingest_for_request(self, ingest_body: IngestBody) -> IngestResponse:

        dispatch = asyncio.create_task(asyncio.to_thread(self.ingest, ingest_body))
        try:
            result = await asyncio.shield(dispatch)
        except asyncio.CancelledError:
            result_val = await asyncio.shield(dispatch)
            logger.info("Celery ingest cancelled after HTTP disconnect")
            del result_val
            raise
        return result

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Convert (DocumentConverter protocol)
    # ------------------------------------------------------------------

    def bytes_to_text(self, raw: bytes, ext: str) -> str:
        """Dispatch parse_task in parse-only mode on the worker, return text."""
        import base64
        import uuid

        from private_gpt.celery.dispatch import dispatch_task
        from private_gpt.celery.tasks.ingestion.extraction_tasks import PARSE_TASK_NAME
        from private_gpt.server.ingest.ingest_router import IngestAsyncBody, IngestBody
        from private_gpt.server.utils.artifact_input import FileArtifact

        config = settings()
        parse_body = IngestAsyncBody(
            ingest_body=IngestBody(
                artifact=f"__convert_{uuid.uuid4().hex}",
                collection="__convert",
                input=FileArtifact(value=base64.b64encode(raw).decode()),
                metadata={"file_name": f"document{ext}"},
            ),
        )
        result = dispatch_task(
            task_name=PARSE_TASK_NAME,
            args=(parse_body,),
            kwargs={"dispatch_store": False},
            queue=config.scheduler.ingestion.celery_queue,
        )
        result_value = wait_for_celery_result(result)
        assert isinstance(result_value, str)
        return result_value


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
        if self._scheduler is not None:
            return self._scheduler

        mode = self._settings.scheduler.ingestion.mode
        provider = _INGESTION_SCHEDULERS.get(mode)
        if provider is None:
            raise ValueError(f"Unknown scheduler.ingestion.mode: {mode}")
        scheduler = cast(
            BaseIngestionScheduler,
            self._injector.get(provider)
            if isinstance(provider, type)
            else provider(self._injector),
        )
        self._scheduler = scheduler
        return scheduler
