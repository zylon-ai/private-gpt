import asyncio
import functools
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec

from private_gpt.chat.input_models import BlobVisibilityMode
from private_gpt.components.storage.s3_helper import S3Helper
from private_gpt.di import get_global_injector
from private_gpt.events.models import BinaryBlock
from private_gpt.events.models._content_blocks import (
    Base64BinarySource,
    URIBinarySource,
)
from private_gpt.settings.settings import settings

logger = logging.getLogger(__name__)

P = ParamSpec("P")


def auto_resolve_media_blocks(
    enabled: bool = True,
    raise_on_error: bool = False,
    blob_visibility: BlobVisibilityMode = BlobVisibilityMode.BINARY,
) -> Callable[[Callable[P, Awaitable[list[Any]]]], Callable[P, Awaitable[list[Any]]]]:
    """Decorator to automatically resolve media blocks based on blob_visibility.

    Modifies blocks in-place to avoid breaking serialization.

    Args:
        enabled: Enable or disable automatic resolution
        raise_on_error: Raise exceptions or log warnings on errors
        blob_visibility: Mode to determine how to handle media data
    """

    def decorator(
        func: Callable[P, Awaitable[list[Any]]]
    ) -> Callable[P, Awaitable[list[Any]]]:
        if not enabled:
            return func

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> list[Any]:
            # Execute the original function
            result = await func(*args, **kwargs)

            # If blob_visibility is BINARY, no transformation needed
            if blob_visibility == BlobVisibilityMode.BINARY:
                return result

            try:
                for block in result:
                    if isinstance(
                        block, BinaryBlock
                    ):  # TODO: now this only supports BinaryBlock
                        # we can consider supporting more block type like image block,
                        # but now the dependency with llama index is not easy to handle,
                        await _transform_media_block_inplace(
                            block=block,
                            blob_visibility=blob_visibility,
                            raise_on_error=raise_on_error,
                        )

                return result

            except Exception as e:
                if raise_on_error:
                    raise
                logger.warning(
                    f"Error transforming media blocks: {e}, returning original result"
                )
                return result

        return async_wrapper

    return decorator


async def _transform_media_block_inplace(
    block: BinaryBlock,
    blob_visibility: BlobVisibilityMode,
    raise_on_error: bool = False,
) -> None:
    """Transform a media block IN PLACE based on visibility mode.

    Modifies the block's source directly to preserve type and serialization.

    Args:
        block: The block to transform (must have source and optional filename attrs)
        blob_visibility: The visibility mode (PUBLIC or PRIVATE)
        raise_on_error: Whether to raise on errors or log and skip
    """
    try:
        # If source is already URI-backed, skip transformation.
        if isinstance(block.source, URIBinarySource) and _is_url(block.source.url):
            return

        if not isinstance(block.source, Base64BinarySource):
            raise ValueError("Binary block must contain base64 data before upload")
        if not block.source.data:
            raise ValueError("Binary block must contain base64 data before upload")

        # Get S3 helper from injector
        s3_helper = get_global_injector().get(S3Helper)

        # Generate unique identifiers
        mime_type = block.source.media_type or "application/octet-stream"
        if block.filename is None:
            block.filename = _generate_filename(mime_type)
        object_name = str(uuid.uuid4())
        bucket_name = settings().s3.temporary_bucket_name

        # Convert base64 data to bytes
        bytes_data = _extract_bytes_from_data(block.source.data)

        # Upload to S3
        s3_url = await asyncio.to_thread(
            s3_helper.upload_file_to_s3,
            bytes_data=bytes_data,
            filename=block.filename,
            bucket_name=bucket_name,
            object_name=object_name,
            mime_type=mime_type,
        )

        # Get appropriate URL based on visibility
        if blob_visibility == BlobVisibilityMode.PUBLIC:
            url = await asyncio.to_thread(
                s3_helper.generate_presigned_download_url,
                s3_url,
            )
        else:  # PRIVATE
            url = s3_url

        block.source = URIBinarySource(type="url", url=url)
    except Exception as e:
        if raise_on_error:
            raise
        logger.warning(
            f"Failed to transform block {type(block).__name__}: {e}, skipping"
        )


def _is_url(data: str) -> bool:
    """Check if data is already a URL."""
    return isinstance(data, str) and (
        data.startswith("http") or data.startswith("s3://")
    )


def _generate_filename(mime_type: str) -> str:
    """Generate a filename based on MIME type."""
    extension = _get_extension(mime_type)
    return f"{uuid.uuid4()}.{extension}"


def _extract_bytes_from_data(data: str) -> bytes:
    """Extract bytes from base64-encoded or string data."""
    import base64

    try:
        # Try to decode as base64 first
        return base64.b64decode(data)
    except Exception:
        # If it fails, encode as UTF-8
        if isinstance(data, str):
            return data.encode("utf-8")
        elif isinstance(data, bytes):
            return data
        else:
            return str(data).encode("utf-8")


def _get_extension(mime_type: str) -> str:
    """Get file extension from MIME type."""
    mime_map = {
        # Text formats
        "text/csv": "csv",
        "application/json": "json",
        "text/plain": "txt",
        # Documents
        "application/pdf": "pdf",
        # Images
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/gif": "gif",
        "image/webp": "webp",
        # Audio
        "audio/mpeg": "mp3",
        "audio/wav": "wav",
        "audio/ogg": "ogg",
        "audio/mp4": "m4a",
        "audio/webm": "webm",
    }
    return mime_map.get(mime_type, "bin")
