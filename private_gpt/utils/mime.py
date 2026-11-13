import mimetypes
from functools import cache
from pathlib import Path

_DEFAULT_MIME_TYPE = "application/octet-stream"


@cache
def is_magic_available() -> bool:
    # python-magic raises ImportError when the native libmagic is missing
    # (stock Windows, minimal Linux images), so an installed wheel is not enough.
    try:
        import magic  # noqa: F401  # ty:ignore[unresolved-import]

        return True
    except ImportError:
        return False


def detect_mime_from_buffer(content: bytes) -> str:
    """Sniff the MIME type of ``content``; octet-stream when libmagic is absent."""
    if not is_magic_available():
        return _DEFAULT_MIME_TYPE
    import magic  # ty:ignore[unresolved-import]

    try:
        return magic.Magic(mime=True).from_buffer(content)
    except Exception:
        return _DEFAULT_MIME_TYPE


def detect_mime_from_file(path: Path) -> str:
    """Sniff the MIME type of ``path``, falling back to its extension."""
    if is_magic_available():
        import magic  # ty:ignore[unresolved-import]

        try:
            return magic.Magic(mime=True).from_file(str(path))
        except Exception:
            return _DEFAULT_MIME_TYPE
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or _DEFAULT_MIME_TYPE
