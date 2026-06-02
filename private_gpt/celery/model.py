from typing import Generic

from celery import Celery
from celery.result import AsyncResult
from pydantic import BaseModel, Field

from private_gpt.utils.custom_typing import T


class Task(BaseModel):
    """Represents an asynchronous task that has been queued."""

    task_id: str = Field(
        ...,
        description="Unique identifier for the asynchronous task, used to track progress and retrieve results",
        examples=["123e4567-e89b-12d3-a456-426614174000", "revoked"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"task_id": "123e4567-e89b-12d3-a456-426614174000"},
                {"task_id": "456e7890-e89b-12d3-a456-426614174001"},
            ]
        }
    }


class TaskStatus(BaseModel, Generic[T]):
    """Represents the current status and result of an asynchronous task operation."""

    task_id: str = Field(
        ...,
        description="Unique identifier of the task being monitored",
        examples=["123e4567-e89b-12d3-a456-426614174000"],
    )
    task_status: str = Field(
        ...,
        description="Current execution status of the task. Common values: PENDING, SUCCESS, FAILURE, REVOKED",
        examples=["PENDING", "SUCCESS", "FAILURE", "REVOKED"],
    )
    task_result: T | str | None = Field(
        ...,
        description="Task execution result. Contains typed result data on success, error message string on failure, or None for pending/deletion tasks",
        examples=[
            None,
            "File format not supported",
            {"object": "list", "model": "private-gpt", "data": []},
        ],
    )

    model_config = {
        "json_schema_extra": {
            "description": "Represents the status of an asynchronous task, including its ID, status, and result.",
            "examples": [
                {
                    "task_id": "123e4567-e89b-12d3-a456-426614174000",
                    "task_status": "SUCCESS",
                    "task_result": {
                        "object": "list",
                        "model": "private-gpt",
                        "data": [
                            {
                                "object": "ingest.document",
                                "artifact": "annual_report",
                                "doc_metadata": {
                                    "file_name": "report.pdf",
                                    "pages": 45,
                                },
                            }
                        ],
                    },
                },
                {
                    "task_id": "456e7890-e89b-12d3-a456-426614174001",
                    "task_status": "PENDING",
                    "task_result": None,
                },
                {
                    "task_id": "789e0123-e89b-12d3-a456-426614174002",
                    "task_status": "FAILURE",
                    "task_result": "URI resource could not be accessed",
                },
            ],
        }
    }

    @classmethod
    def from_celery_task(cls, celery_app: Celery, task_id: str) -> "TaskStatus[T]":
        """Creates a TaskStatus instance from a Celery task result.

        Handles different task states and properly formats results or exceptions
        into the appropriate response structure.
        """
        async_result: AsyncResult[T] = AsyncResult(task_id, app=celery_app)
        # Check if task is in progress, completed or failed
        task_result = (
            async_result.result
            if async_result.status in ("SUCCESS", "FAILURE")
            else None
        )
        if task_result is None:
            # Task in progress, no result yet
            return TaskStatus(
                task_id=task_id, task_status=async_result.status, task_result=None
            )
        elif isinstance(task_result, BaseException):
            # Exception occurred
            return TaskStatus(
                task_id=task_id,
                task_status=async_result.status,
                task_result=str(task_result),
            )
        else:
            # Task completed, typed result is ready
            return TaskStatus(
                task_id=task_id,
                task_status=async_result.status,
                task_result=task_result,
            )
