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
from private_gpt.server.ingest.ingest_router import IngestAsyncBody
from private_gpt.server.utils.artifact_input import UriArtifact
from private_gpt.settings.settings import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if settings().server.debug_mode else logging.INFO)

# Shared callback name: both tasks emit events under this prefix so consumers
# see a unified pgpt.vector_index_task.* stream regardless of which step runs.
VECTOR_INDEX_CALLBACK_TASK_NAME = "vector_index_task"

PARSE_TASK_NAME = "private_gpt.ingestion.parse"
STORE_VECTORS_TASK_NAME = "private_gpt.ingestion.store_vectors"

T = TypeVar("T")

AUTORETRY_EXCEPTIONS = (IndexNotReadyException,)


def cleanup_temporal_files(func: Callable[..., T]) -> Callable[..., T]:
    @wraps(func)
    def wrapper(body: IngestAsyncBody, *args: Any, **kwargs: Any) -> T:
        try:
            result = func(body, *args, **kwargs)
            ensure_to_remove_temporal_files(body)
            return result
        except Exception as e:
            # Since we cannot know if the exception will trigger an auto-retry,
            # we only remove temporal files if the exception is not in the
            # auto-retry list (it will be deleted on the next successful attempt
            # or after the bucket retention period).
            if not isinstance(e, AUTORETRY_EXCEPTIONS):
                ensure_to_remove_temporal_files(body)
            raise

    return wrapper


@celery_app.task(  # ty:ignore[no-matching-overload]
    name=PARSE_TASK_NAME,
    base=StatelessBackgroundTask,
    callback_task_name=VECTOR_INDEX_CALLBACK_TASK_NAME,
    autoretry_for=AUTORETRY_EXCEPTIONS,
)
@cleanup_temporal_files
def parse_task(body: IngestAsyncBody, dispatch_store: bool = True) -> Any:
    """Parse the source file into tree nodes.

    First half of the two-step ingestion pipeline.  Runs atomically:
    validates and parses the file, attaches the resulting nodes to a copy
    of the body, then dispatches ``store_vectors_task`` on the same queue
    and returns its task-id so the caller can poll completion.

    When ``dispatch_store`` is ``False`` the pipeline is used in
    parse-only mode (e.g. chat document conversion): the parsed content is
    returned as plain text and no ``store_vectors_task`` is dispatched.

    Progress and done/error events are published under the
    ``vector_index_task`` callback name so downstream consumers see a
    unified event stream regardless of which task produced them.
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
            f"Parse task for {body.ingest_body.artifact} was skipped. "
            "A delete task is scheduled or running."
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
            f"Ingestion status progress: current-step={status.current_step!s} "
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

    if not dispatch_store:
        # Parse-only mode (chat document conversion): reuse the shared
        # ConvertService logic and return plain text, no store dispatch.
        from private_gpt.server.ingest.convert_service import ConvertService

        convert = ConvertService(service.parse_component)
        extension = get_extension(content.filename) or ""
        return convert.bytes_to_text(content.data.read(), extension)

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

    store_body = body.model_copy(deep=True)
    store_body.nodes = nodes

    from private_gpt.celery.dispatch import dispatch_task

    store_result = dispatch_task(
        task_name=STORE_VECTORS_TASK_NAME,
        args=(store_body,),
        queue=settings().scheduler.ingestion.celery_queue,
    )
    # Return the store_vectors task id so the synchronous caller can poll it.
    return store_result.task_id


@celery_app.task(  # ty:ignore[no-matching-overload]
    name=STORE_VECTORS_TASK_NAME,
    base=StatelessBackgroundTask,
    callback_task_name=VECTOR_INDEX_CALLBACK_TASK_NAME,
    autoretry_for=AUTORETRY_EXCEPTIONS,
)
def store_vectors_task(body: IngestAsyncBody) -> Any:
    """Vectorise pre-parsed nodes and persist them into the vector index.

    Second half of the two-step ingestion pipeline, dispatched automatically
    by ``parse_task``.  Reads the node dicts from ``body.nodes``, rebuilds
    the tree-node objects, and runs the ``load_index`` step — embedding
    generation and vector-store persistence.

    This is the terminal task: its completion triggers the final
    done/error AMQP callback notification to the caller.

    Progress and done/error events are published under the
    ``vector_index_task`` callback name so downstream consumers see a
    unified event stream regardless of which task produced them.
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
            f"Store-vectors task for {body.ingest_body.artifact} was skipped. "
            "A delete task is scheduled or running."
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
            f"Ingestion status progress: current-step={status.current_step!s} "
            f"percentage={status.percentage}, warnings={status.warnings}"
        )
        from private_gpt.celery.callback import run_callback

        run_callback(
            task=store_vectors_task,
            state=custom_states.PROGRESS,
            result=status,
            callback=body.callback,
        )

    nodes = body.nodes or []
    if not nodes:
        return IngestResponse(object="list", model="private-gpt", data=[])

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
        logger.info("Store-vectors task was cancelled, cleaning up")
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
    """Remove temporal files from S3 if the input was a URI."""
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
