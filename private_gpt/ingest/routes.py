from typing import Any

from fastapi import APIRouter, UploadFile

from private_gpt.di import root_injector
from private_gpt.ingest.ingest_service import IngestService

ingest_router = APIRouter()


@ingest_router.post("/ingest")
def ingest(file: UploadFile) -> dict[str, Any]:
    service = root_injector.get(IngestService)
    service.ingest(file.file)
    return {"filename": file.filename}
