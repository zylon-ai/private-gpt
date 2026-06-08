import asyncio
from collections.abc import AsyncIterator
from typing import Literal

from llama_index.core.base.llms.types import MessageRole
from llama_index.core.base.llms.types import TextBlock as LITextBlock
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel

from private_gpt.events.models import DocumentBlock
from private_gpt.server.ingest.convert_service import ConvertService


def _extract_document_blocks(message: ChatMessage) -> list[DocumentBlock]:
    return list(message.additional_kwargs.get("document", []))


class DocumentProcessingStatus(BaseModel):
    status: Literal["processing", "completed", "failed"]
    doc_index: int
    content: str | None = None
    error_detail: str | None = None


class DocumentProcessingResponse(BaseModel):
    message: ChatMessage | None = None
    processing_status: DocumentProcessingStatus | None = None
    chat_history: list[ChatMessage] | None = None


async def _process_document(
    doc_block: DocumentBlock,
    convert_service: ConvertService,
) -> LITextBlock:
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

    return LITextBlock(text="\n\n".join(parts))


async def preprocess_document_message(
    message: ChatMessage,
    convert_service: ConvertService,
    max_concurrency: int = -1,
) -> AsyncIterator[DocumentProcessingResponse]:
    """Preprocess a message by converting DocumentBlock sources in parallel.

    max_concurrency limits how many documents are converted simultaneously.
    -1 (default) means unlimited.
    """
    document_blocks = _extract_document_blocks(message)
    if not document_blocks:
        yield DocumentProcessingResponse(message=message)
        return

    n = len(document_blocks)
    remaining_kwargs = {
        k: v for k, v in message.additional_kwargs.items() if k != "document"
    }

    for i in range(n):
        yield DocumentProcessingResponse(
            processing_status=DocumentProcessingStatus(status="processing", doc_index=i)
        )

    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None

    async def _bounded(doc: DocumentBlock) -> LITextBlock:
        if semaphore is not None:
            async with semaphore:
                return await _process_document(doc, convert_service)
        return await _process_document(doc, convert_service)

    results = await asyncio.gather(
        *[_bounded(doc) for doc in document_blocks],
        return_exceptions=True,
    )

    converted_blocks: list[LITextBlock] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            yield DocumentProcessingResponse(
                processing_status=DocumentProcessingStatus(
                    status="failed",
                    doc_index=i,
                    error_detail=str(result),
                )
            )
            converted_blocks.append(
                LITextBlock(
                    text=(
                        "The user has included document(s) in their message. "
                        "However, we were unable to process the document content. "
                        "Please inform the user that document processing failed."
                    )
                )
            )
        else:
            yield DocumentProcessingResponse(
                processing_status=DocumentProcessingStatus(
                    status="completed",
                    doc_index=i,
                    content=f"Converted document {i + 1} of {n}.",
                )
            )
            converted_blocks.append(result)  # type: ignore[arg-type]

    yield DocumentProcessingResponse(
        message=ChatMessage(
            role=message.role,
            blocks=[*message.blocks, *converted_blocks],
            additional_kwargs=remaining_kwargs,
        )
    )


async def preprocess_document_history(
    chat_history: list[ChatMessage] | None,
    convert_service: ConvertService,
    max_concurrency: int = -1,
) -> AsyncIterator[DocumentProcessingResponse]:
    """Preprocess documents in the last user message only.

    Document blocks are stripped from all other history messages.
    """
    if not chat_history:
        yield DocumentProcessingResponse(chat_history=chat_history)
        return

    if not any(_extract_document_blocks(msg) for msg in chat_history):
        yield DocumentProcessingResponse(chat_history=chat_history)
        return

    if chat_history[-1].role != MessageRole.USER:
        yield DocumentProcessingResponse(chat_history=chat_history)
        return

    is_last_user_message = True
    preprocessed_history: list[ChatMessage] = []

    for message in reversed(chat_history):
        if message.role == MessageRole.USER and is_last_user_message:
            async for response in preprocess_document_message(
                message,
                convert_service=convert_service,
                max_concurrency=max_concurrency,
            ):
                if response.processing_status is not None:
                    yield response
                if response.message is not None:
                    preprocessed_history.append(response.message)
            is_last_user_message = False
        else:
            remaining_kwargs = {
                k: v for k, v in message.additional_kwargs.items() if k != "document"
            }
            preprocessed_history.append(
                ChatMessage(
                    role=message.role,
                    blocks=message.blocks,
                    additional_kwargs=remaining_kwargs,
                )
            )

    yield DocumentProcessingResponse(chat_history=list(reversed(preprocessed_history)))
