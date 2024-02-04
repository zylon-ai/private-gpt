import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, status, Security
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from private_gpt.home import Home
from private_gpt.users import crud, models, schemas
from private_gpt.users.api import deps

from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.ingest.model import IngestedDoc
from private_gpt.server.utils.auth import authenticated
from private_gpt.constants import UPLOAD_DIR

ingest_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])

logger = logging.getLogger(__name__)
class IngestTextBody(BaseModel):
    file_name: str = Field(examples=["Avatar: The Last Airbender"])
    text: str = Field(
        examples=[
            "Avatar is set in an Asian and Arctic-inspired world in which some "
            "people can telekinetically manipulate one of the four elements—water, "
            "earth, fire or air—through practices known as 'bending', inspired by "
            "Chinese martial arts."
        ]
    )


class IngestResponse(BaseModel):
    object: Literal["list"]
    model: Literal["private-gpt"]
    data: list[IngestedDoc]


@ingest_router.post("/ingest", tags=["Ingestion"], deprecated=True)
def ingest(request: Request, file: UploadFile) -> IngestResponse:
    """Ingests and processes a file.

    Deprecated. Use ingest/file instead.
    """
    return ingest_file(request, file)


@ingest_router.post("/ingest/file1", tags=["Ingestion"])
def ingest_file(request: Request, file: UploadFile = File(...)) -> IngestResponse:
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
    service = request.state.injector.get(IngestService)
    if file.filename is None:
        raise HTTPException(400, "No file name provided")
    upload_path = Path(f"{UPLOAD_DIR}/{file.filename}")
    try:
        with open(upload_path, "wb") as f:
            f.write(file.file.read())
        with open(upload_path, "rb") as f:
            ingested_documents = service.ingest_bin_data(file.filename, f)
    except Exception as e:
        return {"message": f"There was an error uploading the file(s)\n {e}"}
    finally:
        file.file.close()
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)
    

@ingest_router.post("/ingest/text", tags=["Ingestion"])
def ingest_text(request: Request, body: IngestTextBody) -> IngestResponse:
    """Ingests and processes a text, storing its chunks to be used as context.

    The context obtained from files is later used in
    `/chat/completions`, `/completions`, and `/chunks` APIs.

    A Document will be generated with the given text. The Document
    ID is returned in the response, together with the
    extracted Metadata (which is later used to improve context retrieval). That ID
    can be used to filter the context used to create responses in
    `/chat/completions`, `/completions`, and `/chunks` APIs.
    """
    service = request.state.injector.get(IngestService)
    if len(body.file_name) == 0:
        raise HTTPException(400, "No file name provided")
    ingested_documents = service.ingest_text(body.file_name, body.text)
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


@ingest_router.get("/ingest/list", tags=["Ingestion"])
def list_ingested(request: Request) -> IngestResponse:
    """Lists already ingested Documents including their Document ID and metadata.

    Those IDs can be used to filter the context used to create responses
    in `/chat/completions`, `/completions`, and `/chunks` APIs.
    """
    service = request.state.injector.get(IngestService)
    ingested_documents = service.list_ingested()
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


@ingest_router.delete("/ingest/{doc_id}", tags=["Ingestion"])
def delete_ingested(request: Request, doc_id: str) -> None:
    """Delete the specified ingested Document.

    The `doc_id` can be obtained from the `GET /ingest/list` endpoint.
    The document will be effectively deleted from your storage context.
    """
    service = request.state.injector.get(IngestService)
    service.delete(doc_id)


@ingest_router.delete("/ingest/file/{filename}", tags=["Ingestion"])
def delete_file(
        request: Request,
        filename: str,
        current_user: models.User = Security(
            deps.get_current_user,
        )) -> dict:
    """Delete the specified filename.

    The `filename` can be obtained from the `GET /ingest/list` endpoint.
    The document will be effectively deleted from your storage context.
    """
    service = request.state.injector.get(IngestService)
    try:
        doc_ids = service.get_doc_ids_by_filename(filename)
        if not doc_ids:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"No documents found with filename '{filename}'")

        for doc_id in doc_ids:
            service.delete(doc_id)

        return {"status": "SUCCESS", "message": f"{filename}' successfully deleted."}
    except Exception as e:
        logger.error(
            f"Unexpected error deleting documents with filename '{filename}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error")


@ingest_router.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
def ingest_file(
        request: Request,
        file: UploadFile = File(...),
        current_user: models.User = Security(
            deps.get_current_user,
        )) -> IngestResponse:
    """Ingests and processes a file, storing its chunks to be used as context."""
    service = request.state.injector.get(IngestService)

    try:
        if file.filename is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="No file name provided")

        upload_path = Path(f"{UPLOAD_DIR}/{file.filename}")

        with open(upload_path, "wb") as f:
            f.write(file.file.read())

        with open(upload_path, "rb") as f:
            ingested_documents = service.ingest_bin_data(file.filename, f)

        return IngestResponse(object="list", model="private-gpt", data=ingested_documents)
    except Exception as e:
        logger.error(f"There was an error uploading the file(s): {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error: Unable to ingest file.",
        )
