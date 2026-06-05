import asyncio
from collections.abc import AsyncIterator
from typing import Literal

from llama_index.core.base.llms.types import TextBlock as LITextBlock
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel

from private_gpt.events.models import DocumentBlock
from private_gpt.server.ingest.convert_service import ConvertService


def _extract_document_blocks(message: ChatMessage) -> list[DocumentBlock]:
    return list(message.additional_kwargs.get("document", []))


class DocumentProcessingStatus(BaseModel):
    status: Literal["processing", "completed", "failed"]
    content: str | None = None
    error_detail: str | None = None


class DocumentProcessingResponse(BaseModel):
    message: ChatMessage | None = None
    processing_status: DocumentProcessingStatus | None = None


async def preprocess_document_message(
    message: ChatMessage,
    convert_service: ConvertService,
) -> AsyncIterator[DocumentProcessingResponse]:
    """Preprocess a message by converting DocumentBlock sources to plain text."""
    document_blocks = _extract_document_blocks(message)
    if not document_blocks:
        yield DocumentProcessingResponse(message=message)
        return

    event = DocumentProcessingStatus(status="processing")
    yield DocumentProcessingResponse(processing_status=event)

    remaining_kwargs = {
        k: v for k, v in message.additional_kwargs.items() if k != "document"
    }
    converted_blocks: list[LITextBlock] = []

    try:
        for doc_block in document_blocks:
            text = await asyncio.to_thread(doc_block.source.to_text, convert_service)

            if not text:
                raise ValueError(
                    f"No content could be extracted from document source "
                    f"(type={doc_block.source.type!r})."
                )

            parts: list[str] = []
            if doc_block.title:
                parts.append(f"Title: {doc_block.title}")
            if doc_block.context:
                parts.append(f"Context: {doc_block.context}")
            parts.append(text)

            converted_blocks.append(LITextBlock(text="\n\n".join(parts)))

        event = event.model_copy(
            update={
                "status": "completed",
                "content": f"Converted {len(document_blocks)} document(s) to text.",
            }
        )
        yield DocumentProcessingResponse(processing_status=event)

    except Exception as e:
        event = event.model_copy(
            update={
                "status": "failed",
                "error_detail": str(e),
            }
        )
        yield DocumentProcessingResponse(processing_status=event)
        converted_blocks = [
            LITextBlock(
                text=(
                    "The user has included document(s) in their message. "
                    "However, we were unable to process the document content. "
                    "Please inform the user that document processing failed."
                )
            )
        ]

    yield DocumentProcessingResponse(
        message=ChatMessage(
            role=message.role,
            blocks=[*message.blocks, *converted_blocks],
            additional_kwargs=remaining_kwargs,
        )
    )
