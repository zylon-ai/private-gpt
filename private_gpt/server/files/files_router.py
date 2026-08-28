from typing import Annotated

from fastapi import APIRouter, Body, Depends, File, Query, Request, UploadFile
from fastapi.responses import Response

from private_gpt.server.files.file_models import (
    DeletedFile,
    DeletedPrefix,
    FileListResponse,
    FileMetadata,
    NamespaceListResponse,
)
from private_gpt.server.files.file_service import FileService
from private_gpt.server.utils.auth import authenticated

files_router = APIRouter(
    prefix="/v1/files",
    dependencies=[Depends(authenticated)],
    tags=["Files"],
    responses={401: {"description": "Unauthorized"}},
)


@files_router.get(
    "/namespaces",
    response_model=NamespaceListResponse,
    summary="List namespaces",
    description=(
        "List all registered filesystem namespaces and their local roots. "
        "Namespaces are configured under `filesystems.namespaces` in settings; "
        "well-known names include 'session' and 'skills'; additional namespaces may be added in settings."
    ),
)
async def list_namespaces(
    request: Request,
) -> NamespaceListResponse:
    service: FileService = request.state.injector.get(FileService)
    return service.list_namespaces()


@files_router.post(
    "",
    response_model=FileMetadata,
    summary="Upload a file",
    description=(
        "Upload into the session filesystem. By default the file goes to the "
        "uploads mount (`/mnt/user-data/uploads/`). A top-level `outputs/` key "
        "selects the deliverables mount (`/mnt/user-data/outputs/`); custom nested "
        "keys such as `data/2024/report.pdf` remain on the uploads mount."
        "The relative path is returned as the file ID. "
        "Uploading a file to an existing key overwrites it."
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
    path: str | None = Query(
        default=None,
        description=(
            "Object-storage-style session key. Defaults to `{filename}` on the "
            "uploads mount. A top-level `outputs/` prefix selects the outputs "
            "mount. Other nested keys are supported and parent directories are "
            "created automatically. Must be relative (no leading `/`), must not "
            "contain `..` components, and must not end with `/`."
        ),
        examples=["data/2024/report.pdf"],
    ),
    namespace: str = Query(
        default="session",
        description="Namespace for the file. Defaults to 'session'.",
        examples=["session"],
    ),
) -> FileMetadata:
    service: FileService = request.state.injector.get(FileService)
    if service._ns_uses_storage(namespace):
        return await service.upload_file(scope_id=scope_id, upload=file, path=path)
    content_bytes = await file.read()
    return await service.write_file_to_namespace(
        namespace=namespace,
        scope=scope_id,
        path=path or file.filename or "upload",
        content=content_bytes,
    )


@files_router.put(
    "/{file_id:path}",
    response_model=FileMetadata,
    summary="Put a file at a specific path (object-storage style)",
    description=(
        "S3/blob-style put-object into the session filesystem. Ordinary keys go to "
        "the uploads mount (`/mnt/user-data/uploads/`). A top-level `outputs/` key "
        "goes to the outputs mount (`/mnt/user-data/outputs/`). Parent "
        "directories are created automatically and existing keys are overwritten. "
        "The response is the same `FileMetadata` as `POST /v1/files`, so the returned "
        "`id` can be used with the other file endpoints."
    ),
    responses={
        400: {"description": "Invalid path key (absolute, `..`, or directory key)."},
        404: {"description": "Namespace not found."},
        503: {"description": "Files API not configured (session namespace not set)."},
    },
)
async def put_file(
    request: Request,
    file_id: str,
    scope_id: str = Query(
        ...,
        description="Session / container identifier.",
        examples=["session-abc123"],
    ),
    namespace: str = Query(
        default="session",
        description="Namespace for the file. Defaults to 'session'.",
        examples=["session"],
    ),
    content: bytes = Body(
        ...,
        description=(
            "Raw file bytes stored at the key (S3/blob-style put-object). "
            "The media type is taken from the request Content-Type header."
        ),
    ),
) -> FileMetadata:
    service: FileService = request.state.injector.get(FileService)
    mime_type = request.headers.get("content-type")
    if service._ns_uses_storage(namespace):
        return await service.put_file(
            scope_id=scope_id, path=file_id, content=content, mime_type=mime_type
        )
    return await service.write_file_to_namespace(
        namespace=namespace,
        scope=scope_id,
        path=file_id,
        content=content,
        mime_type=mime_type,
    )


@files_router.delete(
    "",
    response_model=DeletedPrefix,
    summary="Delete all files matching a prefix (bulk delete)",
    description=(
        "S3/blob-style bulk delete: remove every uploaded file whose key starts "
        "with *prefix* (e.g. `data/2024/` deletes all files in that virtual "
        "folder). Workspace and sandbox output files are deleted individually "
        "through the file-ID endpoint. Returns the count of files actually removed. "
        "Anthropic SDK callers can reach this via "
        "`client.beta.files.with_raw_response` or a plain `httpx` call with "
        "`extra_query={...prefix: data/2024/...}`."
    ),
    responses={
        400: {"description": "prefix is missing or empty."},
        503: {"description": "Files API not configured."},
    },
)
async def delete_by_prefix(
    request: Request,
    scope_id: str = Query(
        ...,
        description="Session / container identifier.",
        examples=["session-abc123"],
    ),
    prefix: str = Query(
        ...,
        description=(
            "Key prefix to delete. All uploaded files whose relative path starts "
            "with this string are removed. Must be non-empty. A trailing `/` is "
            "recommended to avoid accidentally deleting sibling keys that share "
            "the same prefix string (e.g. use `data/2024/` not `data/2024`)."
        ),
        examples=["data/2024/"],
    ),
    namespace: str = Query(
        default="session",
        description="Namespace to delete from. Defaults to 'session'.",
        examples=["session"],
    ),
) -> DeletedPrefix:
    service: FileService = request.state.injector.get(FileService)
    count = await service.delete_prefix(
        scope_id=scope_id, prefix=prefix, namespace=namespace
    )
    return DeletedPrefix(prefix=prefix, deleted_count=count)


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
    namespace: str = Query(
        default="session",
        description="Namespace to list files from. Defaults to 'session'.",
        examples=["session"],
    ),
    prefix: str | None = Query(
        default=None,
        description=(
            "Object-storage-style key prefix filter. When set, only files whose "
            "key starts with this prefix are returned (e.g. `data/2024/` lists "
            "everything uploaded under that virtual folder). Works like S3 "
            "`list-objects-v2 --prefix`. The prefix is matched against the "
            "relative upload key, not the full canonical path."
        ),
        examples=["data/2024/"],
    ),
) -> FileListResponse:
    service: FileService = request.state.injector.get(FileService)
    return await service.list_files(
        scope_id=scope_id,
        limit=limit,
        after_id=after_id,
        before_id=before_id,
        namespace=namespace,
        prefix=prefix,
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
        503: {"description": "Files API not configured (session namespace not set)."},
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
    namespace: str = Query(
        default="session",
        description="Namespace the file belongs to. Defaults to 'session'.",
        examples=["session"],
    ),
    disposition: str = Query(
        default="attachment",
        description=(
            "Content-Disposition mode for the response: `attachment` (default, "
            "browser download) or `inline` (open in browser / preview). "
            "Corresponds to RFC 6266."
        ),
        examples=["attachment"],
    ),
) -> Response:
    service: FileService = request.state.injector.get(FileService)
    content, mime_type, display_name = await service.get_file_content(
        scope_id=scope_id, file_id=file_id, namespace=namespace
    )
    disp = "inline" if disposition.lower() == "inline" else "attachment"
    return Response(
        content=content,
        media_type=mime_type,
        headers={"Content-Disposition": f'{disp}; filename="{display_name}"'},
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
        503: {"description": "Files API not configured (session namespace not set)."},
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
    namespace: str = Query(
        default="session",
        description="Namespace the file belongs to. Defaults to 'session'.",
        examples=["session"],
    ),
) -> FileMetadata:
    service: FileService = request.state.injector.get(FileService)
    return await service.get_file_metadata(
        scope_id=scope_id, file_id=file_id, namespace=namespace
    )


@files_router.head(
    "/{file_id:path}",
    summary="Check file existence and get metadata headers",
    description=(
        "Lightweight existence check and metadata retrieval. Returns the same "
        "metadata as `GET /{file_id}` but as HTTP response headers and with no "
        "body — useful for existence checks and ETag-based conditional requests "
        "without transferring file content. Mirrors S3 `HeadObject` semantics. "
        "Headers returned: `X-File-Id`, `X-File-Mime-Type`, `X-File-Size`, "
        "`X-File-Etag`, `X-File-Namespace`, `X-File-Filename`, "
        "`X-File-Created-At`."
    ),
    responses={
        200: {"description": "File exists; metadata in response headers."},
        404: {"description": "File not found."},
    },
)
async def head_file(
    request: Request,
    file_id: str,
    scope_id: str = Query(
        ...,
        description="Session / container identifier.",
        examples=["session-abc123"],
    ),
    namespace: str = Query(
        default="session",
        description="Namespace the file belongs to. Defaults to 'session'.",
        examples=["session"],
    ),
) -> Response:
    service: FileService = request.state.injector.get(FileService)
    meta = await service.head_file(
        scope_id=scope_id, file_id=file_id, namespace=namespace
    )
    headers = {
        "X-File-Id": meta.id,
        "X-File-Filename": meta.filename,
        "X-File-Mime-Type": meta.mime_type,
        "X-File-Size": str(meta.size_bytes),
        "X-File-Namespace": meta.namespace,
        "X-File-Created-At": meta.created_at.isoformat(),
    }
    if meta.etag:
        headers["X-File-Etag"] = meta.etag
        headers["ETag"] = f'"{meta.etag}"'
    return Response(status_code=200, headers=headers)


@files_router.delete(
    "/{file_id:path}",
    response_model=DeletedFile,
    summary="Delete a file",
    description=(
        "Permanently delete a file from the session. Both uploaded files and "
        "sandbox-generated output files can be deleted through this endpoint."
    ),
    responses={
        404: {"description": "File not found."},
        503: {"description": "Files API not configured (session namespace not set)."},
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
    namespace: str = Query(
        default="session",
        description="Namespace the file belongs to. Defaults to 'session'.",
        examples=["session"],
    ),
) -> DeletedFile:
    service: FileService = request.state.injector.get(FileService)
    return await service.delete_file(
        scope_id=scope_id, file_id=file_id, namespace=namespace
    )
