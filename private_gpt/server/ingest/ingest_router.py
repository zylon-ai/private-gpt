from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.ingest.model import IngestedDoc
from private_gpt.server.utils.auth import authenticated

ingest_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


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
    collection_name: str | None = Field(
        default=None,
        description="Optional collection name to tag the ingested text with.",
        examples=["engineering"],
    )


class IngestResponse(BaseModel):
    object: Literal["list"]
    model: Literal["private-gpt"]
    data: list[IngestedDoc]


@ingest_router.post("/ingest", tags=["Ingestion"], deprecated=True)
def ingest(
    request: Request,
    file: UploadFile,
    collection_name: str | None = Query(default=None),
) -> IngestResponse:
    """Ingests and processes a file.

    Deprecated. Use ingest/file instead.
    """
    return ingest_file(request, file, collection_name=collection_name)


@ingest_router.post("/ingest/file", tags=["Ingestion"])
def ingest_file(
    request: Request,
    file: UploadFile,
    collection_name: str | None = Query(
        default=None,
        description="Tag ingested documents with this collection name.",
    ),
) -> IngestResponse:
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
    ingested_documents = service.ingest_bin_data(
        file.filename, file.file, collection_name=collection_name
    )
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
    ingested_documents = service.ingest_text(
        body.file_name, body.text, collection_name=body.collection_name
    )
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


@ingest_router.get("/ingest/list", tags=["Ingestion"])
def list_ingested(
    request: Request,
    collection_name: str | None = Query(
        default=None,
        description="Filter results to documents in this collection.",
    ),
) -> IngestResponse:
    """Lists already ingested Documents including their Document ID and metadata.

    Those IDs can be used to filter the context used to create responses
    in `/chat/completions`, `/completions`, and `/chunks` APIs.

    Use the optional `collection_name` query parameter to filter by collection.
    """
    service = request.state.injector.get(IngestService)
    all_docs = service.list_ingested()
    if collection_name is not None:
        all_docs = [
            doc
            for doc in all_docs
            if doc.doc_metadata
            and doc.doc_metadata.get("collection_name") == collection_name
        ]
    return IngestResponse(object="list", model="private-gpt", data=all_docs)


class IngestContextBody(BaseModel):
    context_text: str = Field(
        description=(
            "Free-text description of the documents in this collection. "
            "Explain abbreviations, technical codes, and where to find specific "
            "information. For example: 'L1 is the cooler length. L2 is the fan "
            "diameter. TDP is the thermal design power in watts.' "
            "This text is injected into every chunk embedding at ingestion time, "
            "so queries using plain language can match technical abbreviations."
        ),
        examples=["L1 is the cooler length. L2 is the fan diameter. TDP means thermal design power in watts."],
    )
    collection_name: str | None = Field(
        default=None,
        description=(
            "Collection this context applies to. Use null/omit for a global "
            "context that applies to documents ingested without a collection."
        ),
        examples=["datasheets"],
    )


class IngestContextResponse(BaseModel):
    collection_name: str | None
    context_text: str


@ingest_router.post("/ingest/context", tags=["Ingestion"])
def set_ingest_context(
    request: Request, body: IngestContextBody
) -> IngestContextResponse:
    """Set an embedding context descriptor for a collection.

    The context text is stored on disk and automatically injected (as
    embedding-only metadata) into every document chunk when files are
    ingested into the matching collection. This creates semantic connections
    between abbreviations / technical codes and their plain-language
    equivalents so that plain-language queries can retrieve technically
    phrased content.

    **Example use-case:** A datasheet uses "L1" for the cooler length.
    Set a context like *"L1 is the cooler length."* and then ingest the
    datasheet. After that, a query for *"cooler length"* will retrieve the
    chunk containing the L1 value.

    The context applies to future ingestions only. Re-ingest existing
    documents to apply an updated context to them.

    A `collection_name` of `null` (or omitting the field) creates a
    **global** context that acts as a fallback for documents ingested
    without any collection name.
    """
    service = request.state.injector.get(IngestService)
    service.context_manager.save(body.context_text, body.collection_name)
    return IngestContextResponse(
        collection_name=body.collection_name,
        context_text=body.context_text,
    )


@ingest_router.get("/ingest/context", tags=["Ingestion"])
def get_ingest_context(
    request: Request,
    collection_name: str | None = Query(
        default=None,
        description="Collection whose context to retrieve. Omit for the global context.",
    ),
) -> IngestContextResponse:
    """Retrieve the current embedding context descriptor for a collection.

    Returns the context text that will be injected into chunks when
    documents are ingested into the specified collection.
    """
    service = request.state.injector.get(IngestService)
    context_text = service.context_manager.load_exact(collection_name)
    if context_text is None:
        raise HTTPException(
            status_code=404,
            detail=f"No context found for collection={collection_name!r}",
        )
    return IngestContextResponse(
        collection_name=collection_name,
        context_text=context_text,
    )


@ingest_router.delete("/ingest/context", tags=["Ingestion"])
def delete_ingest_context(
    request: Request,
    collection_name: str | None = Query(
        default=None,
        description="Collection whose context to delete. Omit for the global context.",
    ),
) -> None:
    """Delete the embedding context descriptor for a collection.

    After deletion, documents ingested into this collection will no longer
    have contextual metadata injected into their embeddings.
    """
    service = request.state.injector.get(IngestService)
    deleted = service.context_manager.delete(collection_name)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"No context found for collection={collection_name!r}",
        )


@ingest_router.delete("/ingest/{doc_id}", tags=["Ingestion"])
def delete_ingested(request: Request, doc_id: str) -> None:
    """Delete the specified ingested Document.

    The `doc_id` can be obtained from the `GET /ingest/list` endpoint.
    The document will be effectively deleted from your storage context.
    """
    service = request.state.injector.get(IngestService)
    service.delete(doc_id)
