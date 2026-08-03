import json
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

from llama_index.core.schema import BaseNode

from private_gpt.artifact_index.base_artifact_index import IndexNotReadyException
from private_gpt.celery import states as custom_states
from private_gpt.celery.base import StatelessBackgroundTask
from private_gpt.celery.celery import celery_app
from private_gpt.components.ingest.utils import get_extension, get_file_name
from private_gpt.components.storage.s3_helper import S3Helper
from private_gpt.server.ingest.ingest_router import (
    IngestAsyncBody,
)
from private_gpt.server.utils.artifact_input import UriArtifact
from private_gpt.settings.settings import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if settings().server.debug_mode else logging.INFO)

VECTOR_INDEX_TASK_NAME = "private_gpt.ingestion.vector_index"
VECTOR_INDEX_CALLBACK_TASK_NAME = "vector_index_task"

PARSE_TASK_NAME = "private_gpt.ingestion.parse"
PARSE_CALLBACK_TASK_NAME = "parse_task"

EXTRACT_TASK_NAME = "private_gpt.ingestion.extract"
EXTRACT_CALLBACK_TASK_NAME = "extract_task"

T = TypeVar("T")

AUTORETRY_EXCEPTIONS = (IndexNotReadyException,)


def cleanup_temporal_files(func: Callable[..., T]) -> Callable[..., T]:
    @wraps(func)
    def wrapper(body: IngestAsyncBody) -> T:
        try:
            result = func(body)
            ensure_to_remove_temporal_files(body)
            return result
        except Exception as e:
            # Since we cannot know if the exception will trigger an auto-retry,
            # we only remove temporal files if the exception
            # is not in the auto-retry list.
            # Otherwise, it will be deleted on the next
            # successful attempt or after bucket retention period.
            if not isinstance(e, AUTORETRY_EXCEPTIONS):
                ensure_to_remove_temporal_files(body)
            raise

    return wrapper


@celery_app.task(  # ty:ignore[no-matching-overload]
    name=VECTOR_INDEX_TASK_NAME,
    base=StatelessBackgroundTask,
    callback_task_name=VECTOR_INDEX_CALLBACK_TASK_NAME,
    autoretry_for=AUTORETRY_EXCEPTIONS,
)
@cleanup_temporal_files
def vector_index_task(body: IngestAsyncBody) -> Any:
    from private_gpt.celery.notify import ProgressStatus
    from private_gpt.celery.task_helper import IngestionTaskHelper

    # Firstly, we need to check if there is another task
    # that it will roll back the current task.
    from private_gpt.di import get_global_injector
    from private_gpt.server.ingest.ingest_router import IngestResponse
    from private_gpt.server.ingest.ingest_service import IngestService

    if IngestionTaskHelper.is_ingestion_cancel_task_scheduled(
        celery_app=celery_app,
        collection=body.ingest_body.collection,
        artifact=body.ingest_body.artifact,
    ):
        logger.info(
            f"Ingestion task for {body.ingest_body.artifact} was skipped. A delete task is scheduled or running."
        )

        IngestionTaskHelper.revoke_deletion_task(
            celery_app=celery_app,
            collection=body.ingest_body.collection,
            artifact=body.ingest_body.artifact,
        )

        return IngestResponse(
            object="list",
            model="private-gpt",
            data=[],
        )

    def notify(status: ProgressStatus) -> None:
        if body.callback is None:
            return

        logger.debug(
            f"Ingestion status progress: current-step={status.current_step!s} "
            f"percentage={status.percentage}, warnings={status.warnings}"
        )

        from private_gpt.celery.callback import run_callback

        run_callback(
            task=vector_index_task,
            state=custom_states.PROGRESS,
            result=status,
            callback=body.callback,
        )

    service = get_global_injector().get(IngestService)
    content = body.ingest_body.input.to_binary_content(
        filename=get_file_name(body.ingest_body.metadata)
    )
    with service.temporary_file(
        lambda: service.data_path_from_bin_data(
            content.data, get_extension(content.filename)
        )
    ) as file_path:
        try:
            ingested_documents = service.populate_vector_index(
                collection=body.ingest_body.collection,
                artifact=body.ingest_body.artifact,
                file_data=file_path,
                file_metadata=body.ingest_body.metadata,
                notify=notify,
                use_async=settings().data.use_async,
            )
        except SystemExit:
            logger.info("Ingestion task was cancelled, cleaning up")
            # Clean up any partial ingestion if task was cancelled
            service.delete(
                collection=body.ingest_body.collection,
                artifact=body.ingest_body.artifact,
                force=True,  # Force deletion of the index
            )
            raise

    return IngestResponse(
        object="list",
        model="private-gpt",
        data=ingested_documents,
    )


@celery_app.task(  # ty:ignore[no-matching-overload]
    name=PARSE_TASK_NAME,
    base=StatelessBackgroundTask,
    callback_task_name=PARSE_CALLBACK_TASK_NAME,
    autoretry_for=AUTORETRY_EXCEPTIONS,
)
@cleanup_temporal_files
def parse_task(body: IngestAsyncBody) -> Any:
    """Parse a file into nodes and dispatch extract_task as a follow-up.

    This is the first half of the split ingestion pipeline.  It runs
    atomically: validates and parses the source file, serialises the
    resulting nodes into the body, then dispatches extract_task on the
    same queue so that vectorisation happens in a separate step.
    """
    from private_gpt.celery.task_helper import IngestionTaskHelper
    from private_gpt.di import get_global_injector
    from private_gpt.server.ingest.ingest_router import IngestResponse
    from private_gpt.server.ingest.ingest_service import IngestService

    if IngestionTaskHelper.is_ingestion_cancel_task_scheduled(
        celery_app=celery_app,
        collection=body.ingest_body.collection,
        artifact=body.ingest_body.artifact,
    ):
        logger.info(
            f"Parse task for {body.ingest_body.artifact} was skipped. A delete task is scheduled or running."
        )
        IngestionTaskHelper.revoke_deletion_task(
            celery_app=celery_app,
            collection=body.ingest_body.collection,
            artifact=body.ingest_body.artifact,
        )
        return IngestResponse(object="list", model="private-gpt", data=[])

    def notify(status: Any) -> None:
        if body.callback is None:
            return
        logger.debug(
            f"Parse task progress: current-step={status.current_step!s} "
            f"percentage={status.percentage}, warnings={status.warnings}"
        )
        from private_gpt.celery.callback import run_callback

        run_callback(
            task=parse_task,
            state=custom_states.PROGRESS,
            result=status,
            callback=body.callback,
        )

    service = get_global_injector().get(IngestService)
    content = body.ingest_body.input.to_binary_content(
        filename=get_file_name(body.ingest_body.metadata)
    )
    with service.temporary_file(
        lambda: service.data_path_from_bin_data(
            content.data, get_extension(content.filename)
        )
    ) as file_path:
        try:
            file_info, _, warnings = service.parse_component.load_and_validate_file(
                file_data=file_path,
                file_metadata=body.ingest_body.metadata,
                notify=notify,
            )
            nodes = service.ingest_component.parse_file_into_nodes(
                artifact=body.ingest_body.artifact,
                collection=body.ingest_body.collection,
                file_info=file_info,
                file_metadata=body.ingest_body.metadata,
                notify=notify,
                warnings=warnings,
            )
        except SystemExit:
            logger.info("Parse task was cancelled, cleaning up")
            service.delete(
                collection=body.ingest_body.collection,
                artifact=body.ingest_body.artifact,
                force=True,
            )
            raise

    nodes_json = [json.dumps(n.dict()) for n in nodes]

    if body.parse_only:
        return nodes_json

    # Embed serialised nodes into the body and dispatch extract_task on the
    # same queue so the two halves stay together without extra models.
    extract_body = body.model_copy(deep=True)
    extract_body.nodes_json = nodes_json

    from private_gpt.celery.dispatch import dispatch_task

    dispatch_task(
        task_name=EXTRACT_TASK_NAME,
        args=(extract_body,),
        queue=settings().scheduler.ingestion.celery_queue,
    )

    return IngestResponse(object="list", model="private-gpt", data=[])


@celery_app.task(  # ty:ignore[no-matching-overload]
    name=EXTRACT_TASK_NAME,
    base=StatelessBackgroundTask,
    callback_task_name=EXTRACT_CALLBACK_TASK_NAME,
    autoretry_for=AUTORETRY_EXCEPTIONS,
)
def extract_task(body: IngestAsyncBody) -> Any:
    """Vectorise pre-parsed nodes and persist them into the vector index.

    This is the second half of the split ingestion pipeline, dispatched
    automatically by parse_task.  It deserialises the nodes embedded in the
    body and runs the load_index step — embedding generation and vector-store
    persistence — independently from parsing.
    """
    from private_gpt.celery.task_helper import IngestionTaskHelper
    from private_gpt.components.readers.nodes.utils import dict_to_tree_node
    from private_gpt.di import get_global_injector
    from private_gpt.server.ingest.ingest_router import IngestResponse
    from private_gpt.server.ingest.ingest_service import IngestService

    if IngestionTaskHelper.is_ingestion_cancel_task_scheduled(
        celery_app=celery_app,
        collection=body.ingest_body.collection,
        artifact=body.ingest_body.artifact,
    ):
        logger.info(
            f"Extract task for {body.ingest_body.artifact} was skipped. A delete task is scheduled or running."
        )
        IngestionTaskHelper.revoke_deletion_task(
            celery_app=celery_app,
            collection=body.ingest_body.collection,
            artifact=body.ingest_body.artifact,
        )
        return IngestResponse(object="list", model="private-gpt", data=[])

    def notify(status: Any) -> None:
        if body.callback is None:
            return
        logger.debug(
            f"Extract task progress: current-step={status.current_step!s} "
            f"percentage={status.percentage}, warnings={status.warnings}"
        )
        from private_gpt.celery.callback import run_callback

        run_callback(
            task=extract_task,
            state=custom_states.PROGRESS,
            result=status,
            callback=body.callback,
        )

    nodes_json = body.nodes_json or []
    if not nodes_json:
        return IngestResponse(object="list", model="private-gpt", data=[])

    def _split(raw: str) -> tuple[str, str]:
        class_name: str = json.loads(raw).get("class_name", "")
        parts = class_name.rsplit("-", 1)
        return (parts[0], parts[1]) if len(parts) == 2 else (class_name, "v1")

    nodes = [
        dict_to_tree_node(
            version=_split(raw)[1],
            node_type=_split(raw)[0],
            node_dict=json.loads(raw),
        )
        for raw in nodes_json
    ]

    service = get_global_injector().get(IngestService)
    vector_artifact_index = service._make_vector_artifact_index(
        collection=body.ingest_body.collection,
        artifact=body.ingest_body.artifact,
    )

    from llama_index.core import StorageContext, load_index_from_storage

    index = load_index_from_storage(
        index_id=vector_artifact_index.index_id(),
        storage_context=StorageContext.from_defaults(
            vector_store=service.vector_store_component.vector_store(
                body.ingest_body.collection
            ),
            index_store=service.node_store_component.index_store(
                body.ingest_body.collection
            ),
        ),
        embed_model=service.embedding_component.get_embed(),
        transformations=[],
        show_progress=False,
        use_async=False,
        insert_batch_size=512,
    )

    try:
        service.ingest_component.load_index(
            artifact=body.ingest_body.artifact,
            collection=body.ingest_body.collection,
            index=index,
            index_id=vector_artifact_index.index_id(),
            nodes=cast(list[BaseNode], nodes),
            notify=notify,
            use_async=settings().data.use_async,
        )
    except SystemExit:
        logger.info("Extract task was cancelled, cleaning up")
        service.delete(
            collection=body.ingest_body.collection,
            artifact=body.ingest_body.artifact,
            force=True,
        )
        raise

    from private_gpt.server.ingest.model import IngestedDoc

    return IngestResponse(
        object="list",
        model="private-gpt",
        data=[IngestedDoc.from_document(nodes[0])],
    )


def ensure_to_remove_temporal_files(body: IngestAsyncBody) -> None:
    """Remove temporal files from S3 if the input was a URI.

    Since we might have uploaded files to a temporary S3 bucket during ingestion,
    we need to ensure they are removed after the ingestion task is done.
    """
    try:
        from private_gpt.di import get_global_injector

        if isinstance(body.ingest_body.input, UriArtifact):
            temporal_bucket = settings().s3.temporary_bucket_name
            if body.ingest_body.input.is_from_s3_bucket(temporal_bucket):
                uri_value = body.ingest_body.input.value
                logger.info(f"Removing temporary S3 file: {uri_value}")
                s3_helper = get_global_injector().get(S3Helper)
                s3_helper.remove_file_from_s3(uri_value)
    except Exception as e:
        logger.error(f"Failed to remove temporal files: {e}")
