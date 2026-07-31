from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import Response

from private_gpt.server.files.file_models import (
    DeletedFile,
    FileListResponse,
    FileMetadata,
)
from private_gpt.server.files.file_service import FileService
from private_gpt.server.files.namespaced_file_service import NamespacedFileService
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
    after_id: str | None = Query(
        default=None,
        description="Return files created after this file ID (exclusive). Used for forward pagination.",
        examples=["uploads/data.csv"],
    ),
    before_id: str | None = Query(
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


# ---------------------------------------------------------------------------
# Namespace-aware Files API  (T1.3)
# ---------------------------------------------------------------------------

namespaced_files_router = APIRouter(
    prefix="/v1/namespaces",
    dependencies=[Depends(authenticated)],
    tags=["Files (namespaced)"],
    responses={401: {"description": "Unauthorized"}},
)


@namespaced_files_router.post(
    "/{namespace}/{scope:path}/files",
    response_model=FileMetadata,
    summary="Upload a file into a namespace",
    description=(
        "Write a file at ``{namespace}/{scope}/{path}`` where ``path`` is taken from "
        "the uploaded filename.  The caller must supply the ``namespace`` (e.g. "
        "``session``, ``artifacts``) and an opaque ``scope`` identifier (e.g. session "
        "id or org id).  Returns file metadata including etag."
    ),
)
async def ns_upload_file(
    request: Request,
    namespace: str,
    scope: str,
    file: Annotated[UploadFile, File()],
    path: str = Query(
        ...,
        description="Relative path within the scope (e.g. 'outputs/result.png').",
    ),
) -> FileMetadata:
    service: NamespacedFileService = request.state.injector.get(NamespacedFileService)
    return await service.upload_file(
        namespace=namespace, scope=scope, path=path, upload=file
    )


@namespaced_files_router.get(
    "/{namespace}/{scope:path}/files",
    response_model=FileListResponse,
    summary="List files in a namespace scope",
    description="List files under an optional prefix within ``{namespace}/{scope}``.",
)
async def ns_list_files(
    request: Request,
    namespace: str,
    scope: str,
    prefix: str = Query(default="", description="Prefix filter within the scope."),
    limit: int = Query(default=100, ge=1, le=1000),
) -> FileListResponse:
    service: NamespacedFileService = request.state.injector.get(NamespacedFileService)
    return await service.list_by_prefix(
        namespace=namespace, scope=scope, prefix=prefix, limit=limit
    )


@namespaced_files_router.get(
    "/{namespace}/{scope:path}/files/{file_path:path}/content",
    summary="Download file content from a namespace",
    responses={200: {"content": {"application/octet-stream": {}}}},
)
async def ns_get_file_content(
    request: Request,
    namespace: str,
    scope: str,
    file_path: str,
) -> Response:
    service: NamespacedFileService = request.state.injector.get(NamespacedFileService)
    content = await service.read_file(namespace=namespace, scope=scope, path=file_path)
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{file_path.split("/")[-1]}"'
        },
    )


@namespaced_files_router.get(
    "/{namespace}/{scope:path}/files/{file_path:path}",
    response_model=FileMetadata,
    summary="Stat a file in a namespace",
    description="Return metadata (including etag) for a file.",
)
async def ns_stat_file(
    request: Request,
    namespace: str,
    scope: str,
    file_path: str,
) -> FileMetadata:
    service: NamespacedFileService = request.state.injector.get(NamespacedFileService)
    return await service.stat_file(namespace=namespace, scope=scope, path=file_path)


@namespaced_files_router.delete(
    "/{namespace}/{scope:path}/files/{file_path:path}",
    response_model=DeletedFile,
    summary="Delete a file from a namespace",
)
async def ns_delete_file(
    request: Request,
    namespace: str,
    scope: str,
    file_path: str,
) -> DeletedFile:
    service: NamespacedFileService = request.state.injector.get(NamespacedFileService)
    return await service.delete_file(namespace=namespace, scope=scope, path=file_path)
