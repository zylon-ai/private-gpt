import logging
from typing import TYPE_CHECKING

from private_gpt.artifact_index.base_artifact_index import IndexNotReadyException
from private_gpt.celery.base import StatelessBackgroundTask
from private_gpt.celery.celery import celery_app
from private_gpt.celery.task_helper import IngestionTaskHelper
from private_gpt.di import get_global_injector
from private_gpt.settings.settings import settings

if TYPE_CHECKING:
    from private_gpt.server.ingest.ingest_router import (
        DeleteIngestedDocumentAsyncBody,
    )

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if settings().server.debug_mode else logging.INFO)

DELETE_INGESTED_TASK_NAME = "private_gpt.ingestion.delete"
DELETE_INGESTED_CALLBACK_TASK_NAME = "delete_ingested_task"


@celery_app.task(
    name=DELETE_INGESTED_TASK_NAME,
    base=StatelessBackgroundTask,
    # Retry on ValueError and IndexNotReadyException.
    # ValueError is thrown when the index is not initialized
    #   and we cannot guarantee that the index will not be ready.
    # IndexNotReadyException is thrown when the index is being populated.
    autoretry_for=(
        ValueError,
        IndexNotReadyException,
    ),
)
def delete_ingested_task(body: "DeleteIngestedDocumentAsyncBody") -> None:
    from private_gpt.server.ingest.ingest_service import IngestService

    service = get_global_injector().get(IngestService)
    try:
        service.delete(
            collection=body.delete_body.collection,
            artifact=body.delete_body.artifact,
        )
    except (IndexNotReadyException, ValueError):
        # In case that index is not ready, we need to try to
        # revoke the ingestion task, since it's possible that
        # the ingestion task is still running.
        revoked = IngestionTaskHelper.revoke_ingestion_task(
            celery_app=celery_app,
            collection=body.delete_body.collection,
            artifact=body.delete_body.artifact,
        )
        if not revoked:
            # If the task was not revoked, we follow the normal
            # flow and raise the exception.
            raise


delete_ingested_task.callback_task_name = DELETE_INGESTED_CALLBACK_TASK_NAME  # type: ignore[attr-defined]
