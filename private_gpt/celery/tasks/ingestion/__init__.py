from private_gpt.celery.tasks.ingestion.delete_tasks import delete_ingested_task
from private_gpt.celery.tasks.ingestion.extraction_tasks import (
    extract_task,
    parse_task,
    vector_index_task,
)

__all__ = [
    "delete_ingested_task",
    "extract_task",
    "parse_task",
    "vector_index_task",
]
