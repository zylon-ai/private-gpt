import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

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


@celery_app.task(
    name="vector_index_task",
    base=StatelessBackgroundTask,
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
