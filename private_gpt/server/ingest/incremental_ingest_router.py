"""API router for incremental ingestion endpoints.

Exposes REST endpoints for the incremental update pipeline,
mirroring the existing ingest router but with incremental capabilities.

Endpoints:
- POST /v1/incremental-ingest: Ingest a file with incremental updates
- GET  /v1/incremental-ingest/stats: Get update history and statistics
- GET  /v1/incremental-ingest/file-info/{file_name}: Get info about a file
- DELETE /v1/incremental-ingest/{file_name}: Delete a file from the index
"""

import logging
from typing import Any, Literal

from fastapi import APIRouter, Request, UploadFile
from pydantic import BaseModel

from private_gpt.server.ingest.incremental_ingest_service import (
    IncrementalIngestService,
)

logger = logging.getLogger(__name__)

incremental_ingest_router = APIRouter(prefix="/v1", tags=["Incremental Ingestion"])


class IncrementalIngestResponse(BaseModel):
    """Response from an incremental ingest operation."""

    object: Literal["incremental_ingest.result"] = "incremental_ingest.result"
    file_name: str
    total_chunks_old: int = 0
    total_chunks_new: int = 0
    chunks_unchanged: int = 0
    chunks_modified: int = 0
    chunks_added: int = 0
    chunks_deleted: int = 0
    embeddings_computed: int = 0
    embeddings_skipped: int = 0
    efficiency_ratio: float = 0.0
    time_total_s: float = 0.0


class IncrementalStatsResponse(BaseModel):
    """Response with update history statistics."""

    object: Literal["incremental_ingest.stats"] = "incremental_ingest.stats"
    total_updates: int
    updates: list[dict[str, Any]]


class FileInfoResponse(BaseModel):
    """Response with file information."""

    object: Literal["incremental_ingest.file_info"] = "incremental_ingest.file_info"
    info: dict[str, Any]


@incremental_ingest_router.post(
    "/incremental-ingest",
    response_model=IncrementalIngestResponse,
    summary="Ingest a file with incremental updates",
    description=(
        "Upload a file for incremental ingestion. If the file has been "
        "ingested before, only changed chunks are re-embedded and updated "
        "in the vector store. New files are fully ingested."
    ),
)
async def incremental_ingest(
    request: Request,
    file: UploadFile,
) -> IncrementalIngestResponse:
    """Ingest a file using the incremental update pipeline."""
    service = request.state.injector.get(IncrementalIngestService)

    stats = service.incremental_ingest_bin_data(
        file_name=file.filename or "unknown",
        raw_file_data=file.file,
    )

    return IncrementalIngestResponse(
        file_name=stats.file_name,
        total_chunks_old=stats.total_chunks_old,
        total_chunks_new=stats.total_chunks_new,
        chunks_unchanged=stats.chunks_unchanged,
        chunks_modified=stats.chunks_modified,
        chunks_added=stats.chunks_added,
        chunks_deleted=stats.chunks_deleted,
        embeddings_computed=stats.embeddings_computed,
        embeddings_skipped=stats.embeddings_skipped,
        efficiency_ratio=stats.efficiency_ratio,
        time_total_s=stats.time_total_s,
    )


@incremental_ingest_router.get(
    "/incremental-ingest/stats",
    response_model=IncrementalStatsResponse,
    summary="Get incremental update statistics",
    description="Returns the history of all incremental updates in this session.",
)
async def get_incremental_stats(request: Request) -> IncrementalStatsResponse:
    """Get the history of incremental updates."""
    service = request.state.injector.get(IncrementalIngestService)
    history = service.get_update_history()
    return IncrementalStatsResponse(
        total_updates=len(history),
        updates=history,
    )


@incremental_ingest_router.get(
    "/incremental-ingest/file-info/{file_name}",
    response_model=FileInfoResponse,
    summary="Get information about an ingested file",
    description="Returns chunk hash registry information for the given file.",
)
async def get_file_info(request: Request, file_name: str) -> FileInfoResponse:
    """Get information about how a file is stored."""
    service = request.state.injector.get(IncrementalIngestService)
    info = service.get_file_info(file_name)
    return FileInfoResponse(info=info)


@incremental_ingest_router.delete(
    "/incremental-ingest/{file_name}",
    summary="Delete a file from the incremental index",
    description="Removes all chunks for the given file from the vector store and hash registry.",
)
async def delete_incremental_file(request: Request, file_name: str) -> dict[str, Any]:
    """Delete a file and all its chunks."""
    service = request.state.injector.get(IncrementalIngestService)
    success = service.delete_file(file_name)
    return {"deleted": success, "file_name": file_name}
