"""Land attached images and audio in the sandbox uploads mount.

Unlike the document uploader this is purely additive: the blocks stay on the
message. Whether the model can see them natively or they are about to be
replaced by a description is none of this module's business — either way the
bytes also end up on disk, so the model can compute over them with code or
reach for ``describe_image`` / ``transcribe_audio`` when a glance is not enough.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from llama_index.core.base.llms.types import AudioBlock, ImageBlock, MessageRole
from llama_index.core.base.llms.types import TextBlock as LITextBlock
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel

from private_gpt.components.chat.processors.chat_history.multimodality.utils import (
    extract_audio_blocks,
    extract_image_blocks,
)
from private_gpt.components.environment.layout import storage_to_canonical_path
from private_gpt.components.environment.naming import unique_name

if TYPE_CHECKING:
    from private_gpt.server.files.file_service import FileService

logger = logging.getLogger(__name__)


class UploadedMedia(BaseModel):
    """One image or audio attachment after it has been written to uploads."""

    reference: str
    canonical_path: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.canonical_path is not None


def _extension_for(mime: str | None, fallback: str) -> str:
    """Return a file extension for *mime*, falling back when it is unknown."""
    if not mime:
        return fallback
    import filetype  # type: ignore[import-untyped]

    guessed = filetype.get_type(mime=mime)
    return f".{guessed.extension}" if guessed else fallback


def _image_name(block: ImageBlock, index: int) -> str:
    return f"image{index + 1}{_extension_for(block.image_mimetype, '.png')}"


def _audio_name(block: AudioBlock, index: int) -> str:
    # ``format`` is the codec/container ("mp3", "wav"), which is all the wire
    # model carries — there is no filename on an audio block.
    suffix = (block.format or "").strip().lstrip(".")
    return f"audio{index + 1}.{suffix}" if suffix else f"audio{index + 1}.mp3"


def _read_image(block: ImageBlock) -> bytes:
    # ``block.image`` holds base64 *text*, re-encoded at construction. Writing
    # it verbatim would leave an unreadable file on disk, so resolve it back to
    # the real bytes; this also transparently fetches ``path``/``url`` sources.
    return block.resolve_image().read()


def _read_audio(block: AudioBlock) -> bytes:
    return block.resolve_audio().read()


async def _upload_media(
    block: Any,
    name: str,
    reader: Any,
    file_service: "FileService",
    scope_id: str,
    write_lock: asyncio.Lock,
) -> UploadedMedia:
    try:
        # A ``url`` source does blocking I/O, so resolving runs in a thread.
        raw = await asyncio.to_thread(reader, block)
        if not raw:
            raise ValueError("The attachment is empty.")

        # Reading runs in parallel; claiming the name and writing does not, so
        # two attachments cannot both see a name as free and overwrite.
        async with write_lock:
            name = await unique_name(
                name, lambda candidate: file_service.exists(scope_id, candidate)
            )
            await file_service.put_file(scope_id=scope_id, path=name, content=raw)

        return UploadedMedia(
            reference=name,
            canonical_path=storage_to_canonical_path(f"uploads/{name}"),
        )
    except Exception as exc:
        logger.warning("Could not upload media attachment %s", name, exc_info=True)
        return UploadedMedia(reference=name, error=str(exc))


def _build_note(uploads: list[UploadedMedia]) -> str:
    succeeded = [upload for upload in uploads if upload.ok]
    lines = [
        f"The user attached {len(succeeded)} image/audio file(s), "
        "also saved in the sandbox:"
    ]
    lines.extend(f"  {upload.canonical_path}" for upload in succeeded)
    lines.extend(
        f"  {upload.reference} — upload failed: {upload.error}"
        for upload in uploads
        if not upload.ok
    )
    # Saying where the bytes are is the whole point of the note: the model has
    # no other way to learn the paths. It deliberately does not name a tool —
    # what to do with an attachment depends on the request, and naming one here
    # made the model reach for it every time. It does spell out that these are
    # binaries, because a model that can see the image natively will otherwise
    # assume the path is readable as text and waste a turn finding out.
    lines.append(
        "These are binary files. Opening the path as text will not work — "
        "process them with code, or with a tool that accepts a sandbox path."
    )
    return "\n".join(lines)


async def upload_media_message(
    message: ChatMessage,
    file_service: "FileService",
    scope_id: str,
    max_concurrency: int | None = None,
) -> ChatMessage:
    """Persist a message's image/audio attachments, keeping the blocks intact.

    Returns the message unchanged when there is nothing to upload or every
    upload failed — the caller has nothing to fall back to, because the blocks
    were never taken away.
    """
    image_blocks = extract_image_blocks(message)
    audio_blocks = extract_audio_blocks(message)
    if not image_blocks and not audio_blocks:
        return message

    jobs: list[tuple[Any, str, Any]] = [
        (block, _image_name(block, index), _read_image)
        for index, block in enumerate(image_blocks)
    ]
    jobs.extend(
        (block, _audio_name(block, index), _read_audio)
        for index, block in enumerate(audio_blocks)
    )

    semaphore = (
        asyncio.Semaphore(max_concurrency)
        if max_concurrency and max_concurrency > 0
        else None
    )
    write_lock = asyncio.Lock()

    async def _bounded(block: Any, name: str, reader: Any) -> UploadedMedia:
        if semaphore is not None:
            async with semaphore:
                return await _upload_media(
                    block, name, reader, file_service, scope_id, write_lock
                )
        return await _upload_media(
            block, name, reader, file_service, scope_id, write_lock
        )

    uploads = await asyncio.gather(*[_bounded(*job) for job in jobs])

    if not any(upload.ok for upload in uploads):
        return message

    return ChatMessage(
        role=message.role,
        blocks=[*message.blocks, LITextBlock(text=_build_note(uploads))],
        additional_kwargs=dict(message.additional_kwargs),
    )


async def upload_media_history(
    chat_history: list[ChatMessage] | None,
    file_service: "FileService",
    scope_id: str,
    max_concurrency: int | None = None,
) -> list[ChatMessage] | None:
    """Persist the media of the last user message, leaving the rest alone.

    Only the last user message is uploaded, mirroring the document uploader:
    older media is what the multimodal preprocessor trims away, and re-writing
    it on every iteration would fill the mount with copies. Returns ``None``
    when nothing changed, so the caller can skip the state update.
    """
    if not chat_history:
        return None

    last = chat_history[-1]
    if last.role != MessageRole.USER:
        return None

    uploaded = await upload_media_message(
        last,
        file_service=file_service,
        scope_id=scope_id,
        max_concurrency=max_concurrency,
    )
    if uploaded is last:
        return None

    return [*chat_history[:-1], uploaded]
