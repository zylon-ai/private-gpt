from typing import Literal

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from private_gpt.di import root_injector
from private_gpt.server.ingest.ingest_service import IngestedDoc, IngestService

ingest_router = APIRouter(prefix="/v1")


class IngestResponse(BaseModel):
    object: Literal["list"]
    model: Literal["private-gpt"]
    data: list[IngestedDoc]


@ingest_router.post("/ingest", tags=["Ingestion"])
def ingest(file: UploadFile) -> IngestResponse:
    """Ingests and processes a file, storing its chunks to be used as context.

    The context obtained from files is later used in
    `/chat/completions`, `/completions`, and `/chunks` APIs.

    Most common document
    formats are supported, but you may be prompted to install an extra dependency to
    manage a specific file type.

    A file can generate different Documents (for example a PDF generates one Document
    per page). All Documents IDs are returned in the response, together with the
    extracted Metadata (which is later used to improve context retrieval). Those IDs
    can be used to filter the context used to create responses in
    `/chat/completions`, `/completions`, and `/chunks` APIs.
    """
    service = root_injector.get(IngestService)
    if file.filename is None:
        raise HTTPException(400, "No file name provided")
    ingested_documents = service.ingest(file.filename, file.file.read())
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


@ingest_router.get("/ingest/list", tags=["Ingestion"])
def list_ingested() -> IngestResponse:
    """Lists already ingested Documents including their Document ID and metadata.

    Those IDs can be used to filter the context used to create responses
    in `/chat/completions`, `/completions`, and `/chunks` APIs.
    """
    service = root_injector.get(IngestService)
    ingested_documents = service.list_ingested()
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


@ingest_router.delete("/ingest/{doc_id}", tags=["Ingestion"])
def delete_ingested(doc_id: str) -> None:
    """Delete the specified ingested Document.

    The `doc_id` can be obtained from the `GET /ingest/list` endpoint.
    The document will be effectively deleted from your storage context.
    """
    service = root_injector.get(IngestService)
    service.delete(doc_id)
