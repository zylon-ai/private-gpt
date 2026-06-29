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

files_router = APIRouter(
    prefix="/v1/files",
    dependencies=[Depends(authenticated)],
    tags=["Files"],
    responses={401: {"description": "Unauthorized"}},
)


@files_router.post(
    "",
    response_model=FileMetadata,
    summary="Upload a file",
    description=(
        "Upload a file into the session's uploads directory. "
        "The file is stored under `uploads/{filename}` within the session scope and "
        "its relative path is returned as the file ID. "
        "Uploading a file with the same name overwrites the existing one."
    ),
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
        examples=["uploads/data.csv"],
    ),
    before_id: str
    | None = Query(
        default=None,
        description="Return files created before this file ID (exclusive). Used for backward pagination.",
        examples=["outputs/result.png"],
    ),
) -> FileListResponse:
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
        404: {"description": "File not found in the session."},
        503: {"description": "Files API not configured (volume_root not set)."},
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
        "Retrieve metadata for a specific file by its relative path ID, "
        "e.g. `uploads/data.csv` or `outputs/result.png`."
    ),
    responses={
        404: {"description": "File not found in the session."},
        503: {"description": "Files API not configured (volume_root not set)."},
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
        404: {
            "description": "File not found or is a sandbox output (outputs cannot be deleted)."
        },
        503: {"description": "Files API not configured (volume_root not set)."},
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
    service: FileService = request.state.injector.get(FileService)
    return await service.delete_file(scope_id=scope_id, file_id=file_id)
