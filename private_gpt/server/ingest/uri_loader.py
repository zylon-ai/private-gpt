import io
from typing import Any, BinaryIO
from urllib.parse import urlparse

from private_gpt.components.storage.s3_helper import S3Helper
from private_gpt.di import get_global_injector
from private_gpt.settings.settings import settings

DEFAULT_URL_TIMEOUT_SECONDS: tuple[float, float] = (10.0, 60.0)
_DOWNLOAD_CHUNK_SIZE = 64 * 1024


def _load_file_from_url(
    url: str,
    *,
    timeout: float | tuple[float, float] | None = None,
    max_bytes: int | None = None,
    **kwargs: Any,
) -> BinaryIO:
    """Download ``url`` into memory with a timeout, status check and size cap.

    The body is streamed so an oversized response is abandoned as soon as the
    cap is crossed instead of being fully buffered first.
    """
    import requests

    del kwargs
    if timeout is None:
        timeout = DEFAULT_URL_TIMEOUT_SECONDS
    if max_bytes is None:
        max_bytes = settings().chat.maximum_blob_size

    buffer = io.BytesIO()
    total = 0
    with requests.get(url, allow_redirects=True, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(
                    f"Remote file {url} exceeds the maximum allowed size "
                    f"of {max_bytes} bytes."
                )
            buffer.write(chunk)
    buffer.seek(0)
    return buffer


def _load_file_from_base64(base64_str: str, **kwargs: Any) -> BinaryIO:
    import base64

    # Handle data URI format: data:image/png;base64,iVBORw0KG...
    if base64_str.startswith("data:"):
        _, encoded = base64_str.split(",", 1)
        content = base64.b64decode(encoded)
    else:
        content = base64.b64decode(base64_str.strip())

    return io.BytesIO(content)


def _load_file_from_disk(url: str, **kwargs: Any) -> BinaryIO:
    with open(url, "rb") as source:
        content = source.read()
    return io.BytesIO(content)


def _is_url(uri: str) -> bool:
    try:
        result = urlparse(uri)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def _is_base64(uri: str) -> bool:
    import base64

    if uri.startswith("data:"):
        return True

    try:
        base64.b64decode(uri.strip(), validate=True)
        return True
    except Exception:
        return False


def load_file_from_uri(uri: str, **kwargs: Any) -> BinaryIO:
    """Try to understand the type of URI and load the file."""
    if uri.startswith("s3://"):
        # S3 URI
        s3_helper = get_global_injector().get(S3Helper)
        return s3_helper.load_file_from_s3(uri, **kwargs)
    elif _is_url(uri):
        # Public URL
        return _load_file_from_url(uri, **kwargs)
    elif _is_base64(uri):
        # Base64 encoded data
        return _load_file_from_base64(uri, **kwargs)
    else:
        # Default to Local file
        return _load_file_from_disk(uri, **kwargs)
