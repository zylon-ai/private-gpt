import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from private_gpt.server.ingest.ingest_router import DeleteIngestedDocumentAsyncBody
from private_gpt.settings.settings import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if settings().server.debug_mode else logging.INFO)


class TaskInfoStatus:
    ACTIVE = "ACTIVE"
    PENDING = "PENDING"


@dataclass
class TaskInfo:
    task_id: str
    name: str
    status: str
    args: tuple[Any] | None


def find_tasks(
    celery_app: Any, task_name: str, task_status: str | None = None
) -> Iterator[TaskInfo]:
    """Find task matching name and predicate in active tasks."""
    inspector = celery_app.control.inspect()

    if task_status is None or task_status == TaskInfoStatus.ACTIVE:
        active_tasks = inspector.active() or {}
        for worker_tasks in active_tasks.values():
            for task in worker_tasks:
                if task["name"] != task_name:
                    continue

                yield TaskInfo(
                    task_id=task["id"],
                    status=TaskInfoStatus.ACTIVE,
                    name=task["name"],
                    args=task.get("args", None),
                )

    if task_status is None or task_status == TaskInfoStatus.PENDING:
        pending_tasks = inspector.reserved() or {}
        for worker_tasks in pending_tasks.values():
            for task in worker_tasks:
                if task["name"] != task_name:
                    continue

                yield TaskInfo(
                    task_id=task["id"],
                    status=TaskInfoStatus.PENDING,
                    name=task["name"],
                    args=task.get("args", None),
                )


def revoke_task(celery_app: Any, task_id: str) -> None:
    """Revoke task by ID."""
    celery_app.control.revoke(task_id, terminate=True)
    logger.info(f"Revoked task {task_id}")


class IngestionTaskHelper:
    @staticmethod
    def is_ingestion_cancel_task_scheduled(
        celery_app: Any, collection: str, artifact: str
    ) -> bool:
        for task in find_tasks(celery_app, task_name="delete_ingested_task"):
            if not task.args or not isinstance(
                task.args[0], DeleteIngestedDocumentAsyncBody
            ):
                continue

            task_body: DeleteIngestedDocumentAsyncBody = task.args[0]
            if (
                task_body.delete_body.collection == collection
                and task_body.delete_body.artifact == artifact
            ):
                return True

        return False

    @staticmethod
    def revoke_ingestion_task(celery_app: Any, collection: str, artifact: str) -> bool:
        from private_gpt.server.ingest.ingest_router import IngestAsyncBody

        for task in find_tasks(celery_app, task_name="vector_index_task"):
            if not task.args or not isinstance(task.args[0], IngestAsyncBody):
                continue

            task_body = task.args[0]
            if (
                task_body.ingest_body.collection == collection
                and task_body.ingest_body.artifact == artifact
            ):

                revoke_task(celery_app, task.task_id)
                return True

        return False

    @staticmethod
    def revoke_deletion_task(celery_app: Any, collection: str, artifact: str) -> bool:
        for task in find_tasks(celery_app, task_name="delete_ingested_task"):
            if not task.args or not isinstance(
                task.args[0], DeleteIngestedDocumentAsyncBody
            ):
                continue

            task_body: DeleteIngestedDocumentAsyncBody = task.args[0]
            if (
                task_body.delete_body.collection == collection
                and task_body.delete_body.artifact == artifact
            ):
                revoke_task(celery_app, task.task_id)
                return True

        return False
