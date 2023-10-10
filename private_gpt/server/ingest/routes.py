from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from private_gpt.di import root_injector
from private_gpt.server.ingest.ingest_service import IngestedDoc, IngestService

ingest_router = APIRouter(prefix="/v1")


class IngestResponse(BaseModel):
    object: str
    model: str
    documents: list[str]


@ingest_router.post("/ingest")
def ingest(file: UploadFile) -> IngestResponse:
    service = root_injector.get(IngestService)
    if file.filename is None:
        raise HTTPException(400, "No file name provided")
    documents_ids = service.ingest(file.filename, file.file.read())
    return IngestResponse(
        object="document-id", model="private-gpt", documents=documents_ids
    )


@ingest_router.get("/ingest/list")
def list_ingested() -> list[IngestedDoc]:
    service = root_injector.get(IngestService)
    ingested_docs = service.list_ingested()
    return ingested_docs
