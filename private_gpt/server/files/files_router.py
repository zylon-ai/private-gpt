from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import Response

from private_gpt.server.files.file_models import (
    DeletedFile,
    FileListResponse,
    FileMetadata,
)
from private_gpt.server.files.file_service import FileService
from private_gpt.server.utils.auth import authenticated
from private_gpt.settings.settings import settings

files_router = APIRouter(
    prefix="/v1/files",
    dependencies=[Depends(authenticated)],
    tags=["Files"],
    responses={401: {"description": "Unauthorized"}},
)

_FILES_ENABLED = settings().code_execution.volume_root is not None

if _FILES_ENABLED:

    @files_router.post(
        "",
        response_model=FileMetadata,
        summary="Upload a file",
        description=(
            "Upload a file into the session's uploads directory. "
            "The file is stored on the host volume under `uploads/{filename}` and "
            "its absolute path is returned as the file ID. "
            "Uploading a file with the same name overwrites the existing one."
        ),
        responses={
            200: {
                "description": "File uploaded successfully.",
                "content": {
                    "application/json": {
                        "example": {
                            "id": "/local_data/private_gpt/volumes/sessions/session-abc123/uploads/data.csv",
                            "created_at": "2024-01-15T10:30:00Z",
                            "filename": "data.csv",
                            "mime_type": "text/csv",
                            "size_bytes": 4096,
                            "type": "file",
                            "downloadable": False,
                            "scope": {"id": "session-abc123", "type": "session"},
                        }
                    }
                },
            },
            422: {
                "description": "Invalid request parameters.",
                "content": {
                    "application/json": {"example": {"detail": "scope_id is required"}}
                },
            },
        },
        openapi_extra={
            "requestBody": {
                "description": "Multipart form containing the file to upload.",
                "content": {
                    "multipart/form-data": {
                        "examples": {
                            "csv_upload": {
                                "summary": "Upload a CSV file",
                                "value": {"file": "data.csv"},
                            },
                            "image_upload": {
                                "summary": "Upload an image",
                                "value": {"file": "chart.png"},
                            },
                        }
                    }
                },
            }
        },
    )
    async def upload_file(
        request: Request,
        file: Annotated[UploadFile, File()],
        scope_id: str = Query(
            ...,
            description="Session / container identifier (matches the `container` field in chat requests).",
            examples=["session-abc123"],
        ),
    ) -> FileMetadata:
        """Upload a file to the session's uploads directory.

        The file is stored at
        `{volume_root}/sessions/{scope_id}/uploads/{filename}` on the
        host filesystem. The returned `id` is the absolute path and should
        be used verbatim in subsequent requests.
        """
        service: FileService = request.state.injector.get(FileService)
        return await service.upload_file(scope_id=scope_id, upload=file)

    @files_router.get(
        "",
        response_model=FileListResponse,
        summary="List files in a session",
        description=(
            "List all files associated with a session, combining uploaded input files "
            "and sandbox-generated output files. Results are sorted by creation time "
            "and support cursor-based pagination."
        ),
        responses={
            200: {
                "description": "Paginated list of files in the session.",
                "content": {
                    "application/json": {
                        "example": {
                            "data": [
                                {
                                    "id": "/local_data/private_gpt/volumes/sessions/session-abc123/uploads/data.csv",
                                    "created_at": "2024-01-15T10:30:00Z",
                                    "filename": "data.csv",
                                    "mime_type": "text/csv",
                                    "size_bytes": 4096,
                                    "type": "file",
                                    "downloadable": False,
                                    "scope": {
                                        "id": "session-abc123",
                                        "type": "session",
                                    },
                                },
                                {
                                    "id": "/local_data/private_gpt/volumes/sessions/session-abc123/outputs/result.png",
                                    "created_at": "2024-01-15T10:35:00Z",
                                    "filename": "result.png",
                                    "mime_type": "image/png",
                                    "size_bytes": 102400,
                                    "type": "file",
                                    "downloadable": True,
                                    "scope": {
                                        "id": "session-abc123",
                                        "type": "session",
                                    },
                                },
                            ],
                            "first_id": "/local_data/private_gpt/volumes/sessions/session-abc123/uploads/data.csv",
                            "last_id": "/local_data/private_gpt/volumes/sessions/session-abc123/outputs/result.png",
                            "has_more": False,
                        }
                    }
                },
            },
        },
    )
    async def list_files(
        request: Request,
        scope_id: str = Query(
            ...,
            description="Session / container identifier.",
            examples=["session-abc123"],
        ),
        limit: int = Query(
            default=20,
            ge=1,
            le=1000,
            description="Maximum number of files to return per page.",
            examples=[20],
        ),
        after_id: str
        | None = Query(
            default=None,
            description="Return files created after this file ID (exclusive). Used for forward pagination.",
            examples=[
                "/local_data/private_gpt/volumes/sessions/session-abc123/uploads/data.csv"
            ],
        ),
        before_id: str
        | None = Query(
            default=None,
            description="Return files created before this file ID (exclusive). Used for backward pagination.",
            examples=[
                "/local_data/private_gpt/volumes/sessions/session-abc123/outputs/result.png"
            ],
        ),
    ) -> FileListResponse:
        """List all files in a session, merging uploads and sandbox outputs.

        Uploaded files are enumerated from the `uploads/` directory.
        Sandbox output files are enumerated from the `outputs/` directory.
        Both sets are merged and sorted by creation time before pagination is applied.
        """
        service: FileService = request.state.injector.get(FileService)
        return await service.list_files(
            scope_id=scope_id, limit=limit, after_id=after_id, before_id=before_id
        )

    # NOTE: /{file_id:path}/content must be registered before /{file_id:path} so that
    # requests ending in /content are not captured by the more general path route.
    @files_router.get(
        "/{file_id:path}/content",
        summary="Download file content",
        description=(
            "Download the raw binary content of a file. "
            "The response includes an appropriate `Content-Type` header detected via libmagic "
            "and a `Content-Disposition: attachment` header with the original filename."
        ),
        responses={
            200: {
                "description": "Raw file bytes with MIME-typed content.",
                "content": {"application/octet-stream": {}},
            },
            404: {
                "description": "File not found in the session.",
                "content": {
                    "application/json": {
                        "example": {
                            "detail": "File '/local_data/private_gpt/volumes/sessions/session-abc123/uploads/data.csv' not found."
                        }
                    }
                },
            },
        },
    )
    async def get_file_content(
        request: Request,
        file_id: str,
        scope_id: str = Query(
            ...,
            description="Session / container identifier.",
            examples=["session-abc123"],
        ),
    ) -> Response:
        """Stream the raw bytes of a file identified by its absolute path."""
        service: FileService = request.state.injector.get(FileService)
        content, mime_type, display_name = await service.get_file_content(
            scope_id=scope_id, file_id=file_id
        )
        return Response(
            content=content,
            media_type=mime_type,
            headers={"Content-Disposition": f'attachment; filename="{display_name}"'},
        )

    @files_router.get(
        "/{file_id:path}",
        response_model=FileMetadata,
        summary="Get file metadata",
        description=(
            "Retrieve metadata for a specific file by its absolute path ID. "
            "The `file_id` must be URL-encoded when passed as a path segment."
        ),
        responses={
            200: {
                "description": "File metadata.",
                "content": {
                    "application/json": {
                        "example": {
                            "id": "/local_data/private_gpt/volumes/sessions/session-abc123/uploads/data.csv",
                            "created_at": "2024-01-15T10:30:00Z",
                            "filename": "data.csv",
                            "mime_type": "text/csv",
                            "size_bytes": 4096,
                            "type": "file",
                            "downloadable": False,
                            "scope": {"id": "session-abc123", "type": "session"},
                        }
                    }
                },
            },
            404: {
                "description": "File not found in the session.",
                "content": {
                    "application/json": {
                        "example": {
                            "detail": "File '/local_data/private_gpt/volumes/sessions/session-abc123/uploads/data.csv' not found."
                        }
                    }
                },
            },
        },
    )
    async def get_file_metadata(
        request: Request,
        file_id: str,
        scope_id: str = Query(
            ...,
            description="Session / container identifier.",
            examples=["session-abc123"],
        ),
    ) -> FileMetadata:
        """Return metadata for a single file identified by its absolute path."""
        service: FileService = request.state.injector.get(FileService)
        return await service.get_file_metadata(scope_id=scope_id, file_id=file_id)

    @files_router.delete(
        "/{file_id:path}",
        response_model=DeletedFile,
        summary="Delete an uploaded file",
        description=(
            "Permanently delete an uploaded file from the session. "
            "Only files that were uploaded via `POST /v1/files` can be deleted; "
            "sandbox-generated output files cannot be deleted through this endpoint."
        ),
        responses={
            200: {
                "description": "File deleted successfully.",
                "content": {
                    "application/json": {
                        "example": {
                            "id": "/local_data/private_gpt/volumes/sessions/session-abc123/uploads/data.csv",
                            "type": "file_deleted",
                        }
                    }
                },
            },
            404: {
                "description": "File not found or is a sandbox output (outputs cannot be deleted).",
                "content": {
                    "application/json": {
                        "example": {
                            "detail": "File '/local_data/private_gpt/volumes/sessions/session-abc123/uploads/data.csv' not found or is a sandbox output (cannot be deleted)."
                        }
                    }
                },
            },
        },
    )
    async def delete_file(
        request: Request,
        file_id: str,
        scope_id: str = Query(
            ...,
            description="Session / container identifier.",
            examples=["session-abc123"],
        ),
    ) -> DeletedFile:
        """Delete an uploaded file. Returns 404 for sandbox output files."""
        service: FileService = request.state.injector.get(FileService)
        return await service.delete_file(scope_id=scope_id, file_id=file_id)
