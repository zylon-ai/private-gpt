import logging
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from fastapi import APIRouter, Body, Depends, Request
from fastapi.openapi.models import Example
from pydantic import BaseModel, Field

from private_gpt.components.ingestion.ingestion_scheduler import (
    IngestionSchedulerFactory,
)
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.ingest.model import IngestedDoc
from private_gpt.server.utils.artifact_input import IngestableArtifactType
from private_gpt.server.utils.auth import authenticated
from private_gpt.server.utils.callback import BaseCallbackInput
from private_gpt.server.utils.http_disconnect import cancel_on_http_disconnect

try:
    from private_gpt.celery.celery import celery_app
    from private_gpt.celery.model import Task, TaskStatus

    _CELERY_AVAILABLE = True
except ImportError:
    _CELERY_AVAILABLE = False


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ingest_router = APIRouter(
    prefix="/v1/artifacts",
    dependencies=[Depends(authenticated)],
    tags=["Artifacts"],
    responses={401: {"description": "Unauthorized"}},
)

INGEST_REQUEST_EXAMPLES = cast(
    dict[str, Example],
    {
        "file_base64": {
            "summary": "File content (base64)",
            "value": {
                "input": {
                    "type": "file",
                    "value": "JVBERi0xLjQKJaqrrK0KMS...",
                },
                "artifact": "quarterly_report",
                "collection": "financial_docs",
                "metadata": {"file_name": "Q3_Report.pdf"},
            },
        },
        "uri": {
            "summary": "Remote URI",
            "value": {
                "input": {
                    "type": "uri",
                    "value": "s3://company-docs/annual-2023.pdf",
                },
                "artifact": "annual_report_2023",
                "collection": "financial_reports",
                "metadata": {"file_name": "annual-2023.pdf", "year": "2023"},
            },
        },
        "text": {
            "summary": "Plain text content",
            "value": {
                "input": {
                    "type": "text",
                    "value": "Our company was founded in 2020...",
                },
                "artifact": "company_profile",
                "collection": "corporate_docs",
                "metadata": {
                    "file_name": "company_profile.txt",
                    "author": "Marketing Team",
                },
            },
        },
    },
)


class IngestBody(BaseModel):
    """Request body for ingesting content into the system for AI context."""

    artifact: str = Field(
        ...,
        description="Unique identifier for the text being ingested within the collection",
        min_length=1,
        max_length=255,
        examples=["user_manual_chapter_1", "policy_document_2024"],
    )
    collection: str = Field(
        default="pgpt_collection",
        description="Collection name to group related documents for better organization and filtering",
        min_length=1,
        max_length=255,
        examples=["corporate_docs", "user_manuals", "financial_reports"],
    )
    input: IngestableArtifactType = Field(
        ...,
        title="IngestArtifactInput",
        description="Raw input data to be processed and ingested into the system. Can be a file (base64), URI, or plain text",
    )
    metadata: dict[str, Any] | None = Field(
        None,
        description="Optional metadata dictionary containing additional document information. If provided, must include 'file_name' with a valid file extension",
        examples=[
            {
                "file_name": "company_policy.txt",
                "author": "John Doe",
                "department": "HR",
            },
            {
                "file_name": "manual_section.txt",
                "version": "2.1",
                "last_updated": "2024-01-15",
            },
        ],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "artifact": "company_mission_statement",
                "collection": "corporate_docs",
                "input": {
                    "type": "text",
                    "value": (
                        "Our mission is to democratize access to artificial "
                        "intelligence through secure, private, and user-controlled "
                        "AI systems that respect data sovereignty and privacy."
                    ),
                },
                "metadata": {
                    "file_name": "mission_statement.txt",
                    "author": "CEO",
                    "department": "Executive",
                    "created_date": "2024-01-01",
                },
            },
            "examples": [
                {
                    "artifact": "company_mission_statement",
                    "collection": "corporate_docs",
                    "input": {
                        "type": "text",
                        "value": "Our mission is to democratize access to artificial intelligence through secure, private, and user-controlled AI systems that respect data sovereignty and privacy.",
                    },
                    "metadata": {
                        "file_name": "mission_statement.txt",
                        "author": "CEO",
                        "department": "Executive",
                        "created_date": "2024-01-01",
                    },
                },
                {
                    "artifact": "product_description",
                    "collection": "marketing_materials",
                    "input": {
                        "type": "uri",
                        "value": "https://cdn.example.com/reports/annual-2023.pdf",
                    },
                    "metadata": {
                        "file_name": "product_desc.txt",
                        "version": "1.0",
                        "target_audience": "enterprise",
                    },
                },
                {
                    "artifact": "user_manual_chapter_1",
                    "collection": "user_manuals",
                    "input": {
                        "type": "file",
                        "value": (
                            "UEsDBBQABgAIAAAAIQDf3k5bAAAACgAAAHN0cmluZy50eHRVVQ=="
                        ),
                    },
                    "metadata": {
                        "file_name": "chapter_1.txt",
                        "author": "Jane Smith",
                        "created_date": "2024-02-01",
                    },
                },
            ],
        }
    }


class IngestAsyncBody(BaseCallbackInput):
    """Request body for asynchronous URI ingestion."""

    ingest_body: IngestBody = Field(
        ...,
        description="Document ingestion parameters including artifact, collection, input data, and optional metadata",
    )

    model_config = {  # noqa: RUF012
        "json_schema_extra": {
            "examples": [
                {
                    "callback": {
                        "amqp": {
                            "exchange": "ingestion_events",
                            "routing_key_done": "ingest.completed",
                            "routing_key_error": "ingest.failed",
                            "routing_key_progress": "ingest.progress",
                        }
                    },
                    "ingest_body": {
                        "artifact": "large_document_2024",
                        "collection": "technical_docs",
                        "input": {
                            "type": "uri",
                            "value": "https://example.com/docs/large-document.pdf",
                        },
                        "metadata": {
                            "file_name": "large-document.pdf",
                            "size_mb": 45,
                            "pages": 120,
                        },
                    },
                }
            ]
        }
    }


class IngestResponse(BaseModel):
    """Response model for successful document ingestion operations."""

    object: Literal["list"] = Field(
        default="list",
        description="Response object type, always 'list' for ingestion responses",
    )
    model: Literal["private-gpt"] = Field(
        default="private-gpt", description="Model identifier, always 'private-gpt'"
    )
    data: list[IngestedDoc] = Field(
        ...,
        description="List of ingested documents with their metadata and processing information",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "object": "list",
                    "model": "private-gpt",
                    "data": [
                        {
                            "object": "ingest.document",
                            "artifact": "quarterly_report_q1",
                            "doc_metadata": {
                                "file_name": "Q1_Report.pdf",
                                "page_number": 1,
                                "total_pages": 15,
                                "department": "finance",
                            },
                        }
                    ],
                }
            ]
        }
    }


class DeleteIngestedDocumentBody(BaseModel):
    """Request body for deleting specific ingested documents from the system."""

    collection: str = Field(
        ...,
        description="Name of the collection containing the document to be deleted",
        min_length=1,
        max_length=255,
        examples=["financial_reports", "hr_documents"],
    )
    artifact: str = Field(
        ...,
        description="Unique identifier of the document to delete from the specified collection",
        min_length=1,
        max_length=255,
        examples=["quarterly_report_q1", "employee_handbook_2024"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "collection": "financial_reports",
                    "artifact": "q2_2023_report",
                },
                {
                    "collection": "hr_documents",
                    "artifact": "outdated_policy_manual",
                },
            ]
        }
    }


class DeleteIngestedDocumentAsyncBody(BaseCallbackInput):
    """Request body for asynchronous document deletion."""

    delete_body: DeleteIngestedDocumentBody = Field(
        ...,
        description="Document deletion parameters including collection and artifact identifiers",
    )

    model_config = {  # noqa: RUF012
        "json_schema_extra": {
            "examples": [
                {
                    "callback": {
                        "amqp": {
                            "exchange": "deletion_events",
                            "routing_key_done": "delete.completed",
                            "routing_key_error": "delete.failed",
                            "routing_key_progress": "delete.progress",
                        }
                    },
                    "delete_body": {
                        "collection": "financial_reports",
                        "artifact": "obsolete_report_2022",
                    },
                }
            ]
        }
    }


@ingest_router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest Content",
    description="Unified endpoint to ingest files, text, URIs, or already processed content",
    responses={
        200: {
            "description": "Content successfully ingested and processed",
            "content": {
                "application/json": {
                    "example": {
                        "object": "list",
                        "model": "private-gpt",
                        "data": [
                            {
                                "object": "ingest.document",
                                "artifact": "quarterly_report",
                                "doc_metadata": {
                                    "file_name": "Q3_Report.pdf",
                                    "page_number": 1,
                                    "total_pages": 10,
                                },
                            }
                        ],
                    }
                }
            },
        },
        422: {
            "description": "Invalid input format or request parameters",
            "content": {
                "application/json": {"example": {"detail": "Invalid input format"}}
            },
        },
    },
    tags=["Artifacts"],
    openapi_extra={
        "requestBody": {
            "description": "JSON request body supporting multiple input types",
            "content": {
                "application/json": {
                    "examples": INGEST_REQUEST_EXAMPLES,
                }
            },
        },
        "x-fern-examples": [
            {
                "name": "Ingest plain text document",
                "request": {
                    "artifact": "company_profile",
                    "collection": "corporate_docs",
                    "input": {
                        "type": "text",
                        "value": "Our company was founded in 2020...",
                    },
                    "metadata": {
                        "file_name": "company_profile.txt",
                        "author": "Marketing Team",
                    },
                },
                "response": {
                    "body": {
                        "object": "list",
                        "model": "private-gpt",
                        "data": [
                            {
                                "object": "ingest.document",
                                "artifact": "company_profile",
                                "doc_metadata": {
                                    "file_name": "company_profile.txt",
                                    "author": "Marketing Team",
                                },
                            }
                        ],
                    }
                },
            }
        ],
    },
)
async def ingest_content(
    request: Request,
    body: Annotated[
        IngestBody,
        Body(
            description="JSON request body supporting multiple input types",
            openapi_examples=INGEST_REQUEST_EXAMPLES,
        ),
    ],
) -> IngestResponse:
    """Unified endpoint for ingesting content from multiple sources.

    Supports ingestion from:
    - Base64 encoded files
    - Remote URIs (HTTP/HTTPS, S3, etc.)
    - Plain text content
    - Already processed content (ContextFilter objects)

    The ingested content becomes immediately available for use in other API
    endpoints like /messages, /artifacts/search, and /artifacts/content.

    The configured ingestion scheduler decides whether the work runs
    in-process or is dispatched to a worker.
    """
    scheduler = request.state.injector.get(IngestionSchedulerFactory).get()
    result: IngestResponse = await cancel_on_http_disconnect(
        request,
        scheduler.ingest_for_request(body),
    )
    return result


@ingest_router.post(
    "/ingest/async",
    response_model=Task,
    summary="Ingest Content Asynchronously",
    description="Initiate asynchronous ingestion of content from multiple sources",
    responses={
        200: {
            "description": "Successfully initiated ingestion task",
            "content": {
                "application/json": {
                    "example": {"task_id": "123e4567-e89b-12d3-a456-426614174000"}
                }
            },
        },
        422: {
            "description": "Invalid input format or request parameters",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Input format is invalid or parameters are missing"
                    }
                }
            },
        },
        404: {
            "description": "Resource not accessible (for URI inputs)",
            "content": {
                "application/json": {
                    "example": {"detail": "The URI resource could not be accessed"}
                }
            },
        },
    },
    tags=["Artifacts"],
    openapi_extra={
        "requestBody": {
            "description": "JSON request body containing ingestion parameters and callback configuration for asynchronous processing notifications",
            "content": {
                "application/json": {
                    "examples": {
                        "file_async": {
                            "summary": "Async file ingestion",
                            "value": {
                                "callback": {
                                    "amqp": {
                                        "exchange": "ingestion_events",
                                        "routing_key_done": "ingest.completed",
                                        "routing_key_error": "ingest.failed",
                                        "routing_key_progress": "ingest.progress",
                                    }
                                },
                                "ingest_body": {
                                    "input": {
                                        "type": "file",
                                        "value": "JVBERi0xLjQKJaqrrK0KMS...",
                                    },
                                    "artifact": "quarterly_report",
                                    "collection": "financial_docs",
                                    "metadata": {"file_name": "Q3_Report.pdf"},
                                },
                            },
                        },
                        "uri_async": {
                            "summary": "Async URI ingestion",
                            "value": {
                                "callback": {
                                    "amqp": {
                                        "exchange": "ingestion_events",
                                        "routing_key_done": "ingest.completed",
                                        "routing_key_error": "ingest.failed",
                                        "routing_key_progress": "ingest.progress",
                                    }
                                },
                                "ingest_body": {
                                    "input": {
                                        "type": "uri",
                                        "value": "s3://company-docs/annual-2023.pdf",
                                    },
                                    "artifact": "annual_report_2023",
                                    "collection": "financial_reports",
                                    "metadata": {"year": "2023"},
                                },
                            },
                        },
                        "text_async": {
                            "summary": "Async text ingestion",
                            "value": {
                                "callback": {
                                    "amqp": {
                                        "exchange": "ingestion_events",
                                        "routing_key_done": "ingest.completed",
                                        "routing_key_error": "ingest.failed",
                                        "routing_key_progress": "ingest.progress",
                                    }
                                },
                                "ingest_body": {
                                    "input": {
                                        "type": "text",
                                        "value": "Our company was founded in 2020...",
                                    },
                                    "artifact": "company_profile",
                                    "collection": "corporate_docs",
                                    "metadata": {"author": "Marketing Team"},
                                },
                            },
                        },
                    }
                }
            },
        }
    },
)
def ingest_content_async(request: Request, body: IngestAsyncBody) -> Task:
    """Asynchronously process and ingest content from multiple sources.

    This endpoint initiates an asynchronous task to process and ingest content
    from various sources including files, URIs, and text. It's particularly
    useful for large files or when you need non-blocking operations.

    Supported input types:
    - Base64 encoded files
    - Remote URIs (HTTP/HTTPS, S3, etc.)
    - Plain text content
    - Already processed content (ContextFilter objects)

    The context obtained from ingested content is later used in
    `/chat/completions`, `/completions`, and `/artifacts/search` APIs.

    A file can generate different Documents
    (for example a PDF generates one Document
    per page). All Documents are returned in the response, together with the
    extracted Metadata and artifact id,
    which is later used to improve context retrieval
    and can be used to filter the context used to create responses in
    `/chat/completions`, `/completions`,
    `/artifacts/search` and `/artifact/content`.

    Example request:
    ```json
    {
        "callback": {
            "amqp": {
                "exchange": "ingestion",
                "routing_key_done": "ingest.done",
                "routing_key_error": "ingest.error",
                "routing_key_progress": "ingest.progress"
            }
        },
        "ingest_body": {
            "input": {
                "type": "uri",
                "value": "s3://company-docs/annual-2023.pdf"
            },
            "artifact": "annual_report_2023",
            "collection": "financial_reports",
            "metadata": {
                "file_name": "annual_report_2023.pdf",
                "department": "finance",
                "year": "2023"
            }
        }
    }
    ```

    Notes:
    * URIs must be accessible from the server
    * Base64 content should be properly encoded
    * File name with extension is recommended in metadata
    * PDFs generate one document per page
    * Large files are automatically split into chunks
    * Progress can be monitored via /artifacts/ingest/async/{task_id}
    * Resulting context is available in chat/completion APIs
    * Ingested content can be filtered using metadata

    Important to know:
    - Since binary data cannot be passed directly to Celery tasks, files are
      first uploaded to a temporary S3 bucket. The Celery task then retrieves
      the file from S3 for processing. This can incur additional time.

    """
    scheduler = request.state.injector.get(IngestionSchedulerFactory).get()
    try:
        task_id = scheduler.ingest_async(body)
    except NotImplementedError as exc:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=501,
            detail="Async ingestion is not supported by the configured scheduler",
        ) from exc
    return Task(task_id=task_id)


def ingest_data_sync(
    collection: str,
    artifact: str,
    data_path: Path,
    service: IngestService,
    metadata: dict[str, Any] | None,
) -> IngestResponse:
    """Internal helper function for synchronous document ingestion processing.

    Handles the core ingestion workflow including artifact initialization
    and vector index population for both file and text ingestion.
    """
    # Initialize ingestion
    service.initialize_artifact_indices(
        collection=collection,
        artifact=artifact,
    )
    # Populate indexes
    ingested_documents = service.populate_vector_index(
        collection=collection,
        artifact=artifact,
        file_data=data_path,
        file_metadata=metadata,
    )
    return IngestResponse(
        object="list",
        model="private-gpt",
        data=ingested_documents,
    )


if _CELERY_AVAILABLE:

    @ingest_router.get(
        "/ingest/async/{task_id}",
        response_model=TaskStatus[IngestResponse],
        summary="Check Ingestion Task Status",
        description="Retrieve the current status and results of an asynchronous ingestion task",
        tags=["Artifacts"],
        responses={
            200: {
                "description": "Successfully retrieved task status and results",
                "content": {
                    "application/json": {
                        "examples": {
                            "completed_task": {
                                "summary": "Completed ingestion task with results",
                                "value": {
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
                                                    "page_count": 45,
                                                    "department": "finance",
                                                },
                                            }
                                        ],
                                    },
                                },
                            },
                            "pending_task": {
                                "summary": "Task still in progress",
                                "value": {
                                    "task_id": "456e7890-e89b-12d3-a456-426614174001",
                                    "task_status": "PENDING",
                                    "task_result": None,
                                },
                            },
                            "failed_task": {
                                "summary": "Failed ingestion task with error",
                                "value": {
                                    "task_id": "789e0123-e89b-12d3-a456-426614174002",
                                    "task_status": "FAILURE",
                                    "task_result": "File format not supported or URI inaccessible",
                                },
                            },
                        }
                    }
                },
            },
            404: {
                "description": "Task not found or expired",
                "content": {
                    "application/json": {
                        "example": {"detail": "Task with specified ID not found"}
                    }
                },
            },
        },
    )
    def get_ingest_async_status(
        task_id: Annotated[
            str,
            Field(
                ...,
                description="Unique identifier of the ingestion task to check status for",
                examples=["123e4567-e89b-12d3-a456-426614174000"],
            ),
        ],
    ) -> TaskStatus[IngestResponse]:
        """Retrieves the current status and results of an asynchronous ingestion task.

        This endpoint allows monitoring of background ingestion operations initiated
        via the async ingestion endpoints. Task statuses include:

        - PENDING: Task is queued but not yet started
        - SUCCESS: Task completed successfully with results
        - FAILURE: Task failed with error information
        - REVOKED: Task was cancelled before completion
        """
        return TaskStatus[IngestResponse].from_celery_task(
            celery_app=celery_app, task_id=task_id
        )


@ingest_router.get(
    "/list",
    response_model=IngestResponse,
    summary="List Ingested Documents",
    description="Retrieve a list of all documents ingested into a specific collection",
    responses={
        200: {
            "description": "Successfully retrieved document list",
            "content": {
                "application/json": {
                    "example": {
                        "object": "list",
                        "model": "private-gpt",
                        "data": [
                            {
                                "object": "ingest.document",
                                "artifact": "annual_report",
                                "doc_metadata": {
                                    "file_name": "2023_Annual.pdf",
                                    "department": "finance",
                                    "date": "2023-12-31",
                                },
                            },
                            {
                                "object": "ingest.document",
                                "artifact": "policy_manual",
                                "doc_metadata": {
                                    "file_name": "employee_policy.docx",
                                    "department": "hr",
                                    "version": "2.1",
                                },
                            },
                        ],
                    }
                }
            },
        },
        422: {
            "description": "Invalid collection name",
            "content": {
                "application/json": {
                    "example": {"detail": "Collection name is invalid"}
                }
            },
        },
    },
    tags=["Artifacts"],
)
def list_ingested(
    request: Request,
    collection: Annotated[
        str,
        Field(
            default="pgpt_collection",
            description="Name of the collection to list documents from",
            examples=["pgpt_collection", "financial_reports", "hr_documents"],
        ),
    ],
) -> IngestResponse:
    """Retrieves a comprehensive list of all documents ingested.

    This endpoint provides visibility into all available ingested content,
    including document metadata that can be used for filtering in other API
    operations. Essential for understanding what context is available for
    AI interactions.
    """
    service = request.state.injector.get(IngestService)
    ingested_documents = list(service.get_ingested_files(collection))
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


@ingest_router.post(
    "/delete",
    response_model=None,
    summary="Delete Ingested Document",
    description="Remove a specific document and all its associated data from the system",
    responses={
        200: {
            "description": "Document successfully deleted",
            "content": None,
        },
    },
    tags=["Artifacts"],
    openapi_extra={
        "requestBody": {
            "description": "JSON request body specifying the collection and artifact to be deleted from the system",
            "content": {
                "application/json": {
                    "example": {
                        "collection": "financial_reports",
                        "artifact": "q2_2023_report",
                    }
                }
            },
        }
    },
)
def delete_ingested(request: Request, body: DeleteIngestedDocumentBody) -> None:
    """Permanently removes a document and all associated data from the system.

    This operation deletes the document content, metadata, vector embeddings,
    and all related chunks from the specified collection. The deletion is
    immediate and cannot be undone.
    """
    scheduler = request.state.injector.get(IngestionSchedulerFactory).get()
    scheduler.delete(collection=body.collection, artifact=body.artifact)


@ingest_router.post(
    "/delete/async",
    response_model=Task,
    summary="Delete Document Asynchronously",
    responses={
        200: {"model": Task, "description": "Successfully initiated deletion task"},
        422: {"description": "Invalid request parameters"},
    },
    tags=["Artifacts"],
    openapi_extra={
        "requestBody": {
            "description": "JSON request body containing deletion parameters and optional callback configuration for asynchronous processing notifications",
            "content": {
                "application/json": {
                    "example": {
                        "callback": {
                            "amqp": {
                                "exchange": "deletion_events",
                                "routing_key_done": "delete.completed",
                                "routing_key_error": "delete.failed",
                                "routing_key_progress": "delete.progress",
                            }
                        },
                        "delete_body": {
                            "collection": "financial_reports",
                            "artifact": "obsolete_report_2022",
                        },
                    }
                }
            },
        }
    },
)
def delete_ingested_async(
    request: Request, body: DeleteIngestedDocumentAsyncBody
) -> Task:
    """Initiates asynchronous deletion of a document and all associated data.

    This endpoint queues a deletion task for background processing, making it
    suitable for large documents or when non-blocking operation is required.
    The task can be monitored using the returned task ID.

    If an ingestion task is currently running for the same document, it will
    be automatically revoked before initiating the deletion.
    """
    scheduler = request.state.injector.get(IngestionSchedulerFactory).get()
    try:
        task_id = scheduler.delete_async(body)
    except NotImplementedError as exc:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=501,
            detail="Async deletion is not supported by the configured scheduler",
        ) from exc
    return Task(task_id=task_id)


if _CELERY_AVAILABLE:

    @ingest_router.get(
        "/delete/async/{task_id}",
        response_model=TaskStatus[None],
        tags=["Artifacts"],
        summary="Check Deletion Task Status",
        description="Retrieve the current status of an asynchronous deletion task",
        responses={
            200: {
                "description": "Successfully retrieved task status",
                "content": {
                    "application/json": {
                        "examples": {
                            "completed_deletion": {
                                "summary": "Completed deletion task",
                                "value": {
                                    "task_id": "123e4567-e89b-12d3-a456-426614174000",
                                    "task_status": "SUCCESS",
                                    "task_result": None,
                                },
                            },
                            "pending_deletion": {
                                "summary": "Deletion task in progress",
                                "value": {
                                    "task_id": "456e7890-e89b-12d3-a456-426614174001",
                                    "task_status": "PENDING",
                                    "task_result": None,
                                },
                            },
                            "failed_deletion": {
                                "summary": "Failed deletion task",
                                "value": {
                                    "task_id": "789e0123-e89b-12d3-a456-426614174002",
                                    "task_status": "FAILURE",
                                    "task_result": "Document not found in collection",
                                },
                            },
                        }
                    }
                },
            },
            404: {
                "description": "Task not found or expired",
                "content": {
                    "application/json": {
                        "example": {"detail": "Task with specified ID not found"}
                    }
                },
            },
        },
    )
    def get_delete_async_status(
        task_id: Annotated[
            str,
            Field(
                ...,
                description="Unique identifier of the deletion task to check status for",
                examples=["123e4567-e89b-12d3-a456-426614174000"],
            ),
        ],
    ) -> TaskStatus[None]:
        """Retrieves the current status of an asynchronous deletion task.

        This endpoint allows monitoring of background deletion operations initiated
        via the async deletion endpoint. Task statuses include:

        - PENDING: Task is queued but not yet started
        - SUCCESS: Task completed successfully (task_result is None)
        - FAILURE: Task failed with error information in task_result
        - REVOKED: Task was cancelled before completion
        """
        return TaskStatus[None].from_celery_task(celery_app=celery_app, task_id=task_id)
