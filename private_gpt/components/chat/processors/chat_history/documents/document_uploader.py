import asyncio
import logging
from pathlib import PurePosixPath

from llama_index.core.base.llms.types import MessageRole
from llama_index.core.base.llms.types import TextBlock as LITextBlock
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel

from private_gpt.components.chat.processors.chat_history.documents.document_preprocessor import (
    extract_document_blocks,
)
from private_gpt.components.environment.layout import storage_to_canonical_path
from private_gpt.components.environment.naming import unique_name
from private_gpt.events.models import DocumentBlock
from private_gpt.events.models._content_blocks import DocumentConverter
from private_gpt.server.files.file_service import FileService

logger = logging.getLogger(__name__)


class UploadedDocument(BaseModel):
    """One attachment after it has been written to the session uploads mount."""

    reference: str
    canonical_path: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.canonical_path is not None


def _filename_for(block: DocumentBlock, index: int) -> str:
    """Derive a safe single-segment upload key for *block*."""
    raw = (block.title or "").strip()
    name = PurePosixPath(raw).name if raw else ""
    if not name or name in {".", ".."}:
        name = f"document{index + 1}"

    if not PurePosixPath(name).suffix:
        extension = getattr(block.source, "extension", None)
        name = f"{name}{extension() if callable(extension) else '.txt'}"

    return name


def _bytes_for(block: DocumentBlock, convert_service: DocumentConverter) -> bytes:
    """Return the raw bytes to persist for *block*.

    Text-only sources carry no binary payload, so their text is stored as-is.
    """
    to_bytes = getattr(block.source, "to_bytes", None)
    if callable(to_bytes):
        return bytes(to_bytes())
    return block.source.to_text(convert_service).encode()


async def _upload_document(
    block: DocumentBlock,
    index: int,
    file_service: FileService,
    scope_id: str,
    convert_service: DocumentConverter,
    write_lock: asyncio.Lock,
) -> UploadedDocument:
    reference = block.title or f"document{index + 1}"
    try:
        raw = await asyncio.to_thread(_bytes_for, block, convert_service)
        if not raw:
            raise ValueError("The document is empty.")

        # Reading the bytes runs in parallel; claiming the name and writing does
        # not. Two attachments sharing a title would otherwise both see the name
        # as free and the second would overwrite the first instead of taking the
        # ``(2)`` suffix.
        name = _filename_for(block, index)
        async with write_lock:
            name = await unique_name(
                name, lambda candidate: file_service.exists(scope_id, candidate)
            )
            await file_service.put_file(scope_id=scope_id, path=name, content=raw)

        return UploadedDocument(
            reference=reference,
            canonical_path=storage_to_canonical_path(f"uploads/{name}"),
        )
    except Exception as exc:
        logger.warning("Could not upload attachment %s", reference, exc_info=True)
        return UploadedDocument(reference=reference, error=str(exc))


def _build_note(uploads: list[UploadedDocument]) -> str:
    succeeded = [upload for upload in uploads if upload.ok]
    lines = [
        f"The user attached {len(succeeded)} file(s), now available in the sandbox:"
    ]
    lines.extend(f"  {upload.canonical_path}" for upload in succeeded)
    lines.extend(
        f"  {upload.reference} — upload failed: {upload.error}"
        for upload in uploads
        if not upload.ok
    )
    lines.append(
        "\nUse convert_documents to convert them to markdown in the workspace, or "
        "work with them directly (bash, pandas, …)."
    )
    return "\n".join(lines)


async def upload_document_message(
    message: ChatMessage,
    file_service: FileService,
    scope_id: str,
    convert_service: DocumentConverter,
    max_concurrency: int | None = None,
) -> ChatMessage | None:
    """Persist a message's attachments and replace them with a path listing.

    Returns ``None`` when every upload failed, so the caller can fall back to
    inline conversion rather than send the model a note about nothing.
    """
    document_blocks = extract_document_blocks(message)
    if not document_blocks:
        return message

    semaphore = (
        asyncio.Semaphore(max_concurrency)
        if max_concurrency and max_concurrency > 0
        else None
    )
    write_lock = asyncio.Lock()

    async def _bounded(block: DocumentBlock, index: int) -> UploadedDocument:
        if semaphore is not None:
            async with semaphore:
                return await _upload_document(
                    block, index, file_service, scope_id, convert_service, write_lock
                )
        return await _upload_document(
            block, index, file_service, scope_id, convert_service, write_lock
        )

    uploads = await asyncio.gather(
        *[_bounded(block, index) for index, block in enumerate(document_blocks)]
    )

    if not any(upload.ok for upload in uploads):
        return None

    return ChatMessage(
        role=message.role,
        blocks=[*message.blocks, LITextBlock(text=_build_note(uploads))],
        additional_kwargs={
            k: v
            for k, v in message.additional_kwargs.items()
            if k not in {"document", "binary"}
        },
    )


async def upload_document_history(
    chat_history: list[ChatMessage] | None,
    file_service: FileService,
    scope_id: str,
    convert_service: DocumentConverter,
    max_concurrency: int | None = None,
) -> list[ChatMessage] | None:
    """Upload the attachments of the last user message only.

    Document blocks are stripped from every other history message, mirroring
    ``preprocess_document_history``. Returns ``None`` when the uploads could not
    be performed at all, so the caller falls back to inline conversion.
    """
    if not chat_history:
        return chat_history

    if not any(extract_document_blocks(msg) for msg in chat_history):
        return chat_history

    if chat_history[-1].role != MessageRole.USER:
        return chat_history

    is_last_user_message = True
    uploaded_history: list[ChatMessage] = []

    for message in reversed(chat_history):
        if message.role == MessageRole.USER and is_last_user_message:
            uploaded = await upload_document_message(
                message,
                file_service=file_service,
                scope_id=scope_id,
                convert_service=convert_service,
                max_concurrency=max_concurrency,
            )
            if uploaded is None:
                return None
            uploaded_history.append(uploaded)
            is_last_user_message = False
        else:
            uploaded_history.append(
                ChatMessage(
                    role=message.role,
                    blocks=message.blocks,
                    additional_kwargs={
                        k: v
                        for k, v in message.additional_kwargs.items()
                        if k not in {"document", "binary"}
                    },
                )
            )

    return list(reversed(uploaded_history))
