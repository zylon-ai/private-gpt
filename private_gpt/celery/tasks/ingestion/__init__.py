from private_gpt.celery.tasks.ingestion.delete_tasks import delete_ingested_task
from private_gpt.celery.tasks.ingestion.extraction_tasks import vector_index_task

__all__ = [
    "delete_ingested_task",
    "vector_index_task",
]
