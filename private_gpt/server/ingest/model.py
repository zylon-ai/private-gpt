from typing import Any, Literal

from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore.types import RefDocInfo
from pydantic import BaseModel, Field

from private_gpt.components.ingest.metadata_helper import MetadataKeys


class IngestedDoc(BaseModel):
    """Represents a document that has been ingested into the system."""

    object: Literal["ingest.document"] = Field(
        default="ingest.document",
        description="Type of the object, indicating this is an ingested document.",
    )
    artifact: str = Field(
        description="Unique identifier for the ingested document artifact.",
        examples=["artifact_id_1"],
    )
    doc_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Metadata associated with the ingested document, such as title, author, and other",
        examples=[
            {
                "title": "Sales Report Q3 2023",
                "file_name": "Sales Report Q3 2023.pdf",
            }
        ],
    )

    @staticmethod
    def from_document(document: BaseNode) -> "IngestedDoc":
        return IngestedDoc(
            object="ingest.document",
            artifact=str(document.metadata.get(MetadataKeys.ARTIFACT_ID.value)),
            doc_metadata=document.metadata,
        )

    @staticmethod
    def from_ref_doc_info(ref_doc_info: RefDocInfo) -> "IngestedDoc":
        return IngestedDoc(
            object="ingest.document",
            artifact=str(ref_doc_info.metadata.get(MetadataKeys.ARTIFACT_ID.value)),
            doc_metadata=ref_doc_info.metadata if ref_doc_info else None,
        )
