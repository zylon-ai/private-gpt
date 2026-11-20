import mimetypes
import sys
from functools import cache
from pathlib import Path

_DEFAULT_MIME_TYPE = "application/octet-stream"


@cache
def is_magic_available() -> bool:
    # python-magic on Windows loads whatever magic DLL is on PATH (e.g. Git's
    # MSYS build) and from_buffer/from_file can block forever.
    if sys.platform == "win32":
        return False
    # python-magic raises ImportError when the native libmagic is missing
    # (minimal Linux images, macOS without Homebrew's libmagic).
    try:
        import magic  # noqa: F401  # ty:ignore[unresolved-import]

        return True
    except ImportError:
        return False


def _sniff(head: bytes) -> str | None:
    import filetype  # type: ignore[import-untyped]

    kind = filetype.guess(head)
    return kind.mime if kind is not None else None


def detect_mime_from_buffer(content: bytes) -> str:
    """Sniff the MIME type of ``content``; octet-stream when unknown."""
    if not is_magic_available():
        return _sniff(content[:8192]) or _DEFAULT_MIME_TYPE
    import magic  # ty:ignore[unresolved-import]

    try:
        return magic.Magic(mime=True).from_buffer(content)
    except Exception:
        return _DEFAULT_MIME_TYPE


def detect_mime_from_file(path: Path) -> str:
    """Sniff the MIME type of ``path``, falling back to its extension."""
    detected = detect_mime_from_file_content(path)
    if detected is not None:
        return detected
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or _DEFAULT_MIME_TYPE


def detect_mime_from_file_content(path: Path) -> str | None:
    """Sniff the MIME type of ``path`` from its bytes; None when unknown."""
    if is_magic_available():
        import magic  # ty:ignore[unresolved-import]

        try:
            return magic.Magic(mime=True).from_file(str(path))
        except Exception:
            return None
    try:
        with path.open("rb") as fh:
            return _sniff(fh.read(8192))
    except OSError:
        return None
