from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, UploadFile

from private_gpt.di import root_injector
from private_gpt.server.ingest.ingest_service import IngestService, IngestedDoc

ingest_router = APIRouter(prefix="/v1")


@dataclass
class IngestResponse:
    object: str
    model: str
    documents: list[str]


@ingest_router.post("/ingest")
def ingest(file: UploadFile) -> IngestResponse:
    service = root_injector.get(IngestService)
    documents_ids = service.ingest(file.file)
    return IngestResponse(
        object="document-id", model="private-gpt", documents=documents_ids
    )


@ingest_router.get("/ingest/list")
def list_ingested() -> list[IngestedDoc]:
    service = root_injector.get(IngestService)
    ingested_docs = service.list_ingested()
    return ingested_docs
