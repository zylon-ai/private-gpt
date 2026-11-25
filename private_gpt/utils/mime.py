import io
import mimetypes
import sys
import zipfile
import zlib
from functools import cache
from pathlib import Path

_DEFAULT_MIME_TYPE = "application/octet-stream"

# libmagic's OOXML rule only inspects the first few ZIP members, and filetype
# stops at the PK header. Word often writes ``[trash]/`` parts first, so both
# report ``application/zip`` for a real document.
_ZIP_MIME_TYPES = frozenset(
    {
        "application/zip",
        "application/x-zip",
        "application/x-zip-compressed",
    }
)
_MAX_PACKAGE_MEMBERS = 4096
_MAX_CONTENT_TYPES_BYTES = 1_048_576
_MAX_PACKAGE_MIMETYPE_BYTES = 256

_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_DOTX = "application/vnd.openxmlformats-officedocument.wordprocessingml.template"
_DOCM = "application/vnd.ms-word.document.macroEnabled.12"
_DOTM = "application/vnd.ms-word.template.macroEnabled.12"
_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_XLTX = "application/vnd.openxmlformats-officedocument.spreadsheetml.template"
_XLSM = "application/vnd.ms-excel.sheet.macroEnabled.12"
_XLTM = "application/vnd.ms-excel.template.macroEnabled.12"
_XLAM = "application/vnd.ms-excel.addin.macroEnabled.12"
_XLSB = "application/vnd.ms-excel.sheet.binary.macroEnabled.12"
_PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
_POTX = "application/vnd.openxmlformats-officedocument.presentationml.template"
_PPSX = "application/vnd.openxmlformats-officedocument.presentationml.slideshow"
_PPTM = "application/vnd.ms-powerpoint.presentation.macroEnabled.12"
_POTM = "application/vnd.ms-powerpoint.template.macroEnabled.12"
_PPSM = "application/vnd.ms-powerpoint.slideshow.macroEnabled.12"

# Part content types from [Content_Types].xml, matched as bytes so a broken
# XML declaration cannot hide the package type.
_CONTENT_TYPE_MARKERS: tuple[tuple[bytes, str], ...] = (
    (b"application/vnd.ms-word.document.macroenabled.main+xml", _DOCM),
    (b"application/vnd.ms-word.template.macroenabled.main+xml", _DOTM),
    (
        b"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
        _DOCX,
    ),
    (
        b"application/vnd.openxmlformats-officedocument.wordprocessingml.template.main+xml",
        _DOTX,
    ),
    (b"application/vnd.ms-excel.sheet.binary.macroenabled.main+xml", _XLSB),
    (b"application/vnd.ms-excel.sheet.macroenabled.main+xml", _XLSM),
    (b"application/vnd.ms-excel.template.macroenabled.main+xml", _XLTM),
    (b"application/vnd.ms-excel.addin.macroenabled.main+xml", _XLAM),
    (
        b"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml",
        _XLSX,
    ),
    (
        b"application/vnd.openxmlformats-officedocument.spreadsheetml.template.main+xml",
        _XLTX,
    ),
    (b"application/vnd.ms-powerpoint.presentation.macroenabled.main+xml", _PPTM),
    (b"application/vnd.ms-powerpoint.template.macroenabled.main+xml", _POTM),
    (b"application/vnd.ms-powerpoint.slideshow.macroenabled.main+xml", _PPSM),
    (
        b"application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml",
        _PPTX,
    ),
    (
        b"application/vnd.openxmlformats-officedocument.presentationml.template.main+xml",
        _POTX,
    ),
    (
        b"application/vnd.openxmlformats-officedocument.presentationml.slideshow.main+xml",
        _PPSX,
    ),
)

_PACKAGE_PARTS: tuple[tuple[str, str], ...] = (
    ("word/document.xml", _DOCX),
    ("xl/workbook.bin", _XLSB),
    ("xl/workbook.xml", _XLSX),
    ("ppt/presentation.xml", _PPTX),
)
# Sniffers label every word/ package as docx. Re-read those too, so a macro
# workbook is not left as the generic type the first ZIP member suggested.
_OOXML_MIME_TYPES = frozenset(package_mime for _, package_mime in _CONTENT_TYPE_MARKERS)


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


def _mime_base(value: str) -> str:
    return value.split(";", 1)[0].strip().lower()


def _starts_with_zip(source: bytes | Path) -> bool:
    if isinstance(source, bytes):
        return source.startswith(b"PK\x03\x04")
    try:
        with source.open("rb") as fh:
            return fh.read(4) == b"PK\x03\x04"
    except OSError:
        return False


def refine_detected_mime(detected: str, source: bytes | Path) -> str:
    """Return the Office or ODF type when a sniffer only saw a ZIP container."""
    base = _mime_base(detected)
    should_open = (
        base in _ZIP_MIME_TYPES
        or base in _OOXML_MIME_TYPES
        or base.startswith("application/vnd.oasis.opendocument.")
        or base == "application/epub+zip"
        or (base == _DEFAULT_MIME_TYPE and _starts_with_zip(source))
    )
    if not should_open:
        return detected
    return _office_package_mime(source) or detected


def _open_zip(source: bytes | Path) -> zipfile.ZipFile | None:
    try:
        if isinstance(source, bytes):
            return zipfile.ZipFile(io.BytesIO(source))
        return zipfile.ZipFile(source)
    except (zipfile.BadZipFile, OSError, ValueError):
        return None


def _office_package_mime(source: bytes | Path) -> str | None:
    archive = _open_zip(source)
    if archive is None:
        return None
    try:
        names, content_types, odf_mimetype = _package_entries(archive)
        package_mime = _mime_from_content_types(archive, content_types)
        if package_mime is not None:
            return package_mime
        for part, part_mime in _PACKAGE_PARTS:
            if part in names:
                return part_mime
        return _mime_from_odf_member(archive, odf_mimetype)
    except (zipfile.BadZipFile, OSError, ValueError, RuntimeError, zlib.error):
        return None
    finally:
        archive.close()


def _package_entries(
    archive: zipfile.ZipFile,
) -> tuple[set[str], zipfile.ZipInfo | None, zipfile.ZipInfo | None]:
    names: set[str] = set()
    content_types: zipfile.ZipInfo | None = None
    odf_mimetype: zipfile.ZipInfo | None = None
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        names.add(name)
        if name == "[Content_Types].xml":
            content_types = info
        elif name == "mimetype":
            odf_mimetype = info
        if len(names) >= _MAX_PACKAGE_MEMBERS:
            break
    return names, content_types, odf_mimetype


def _mime_from_content_types(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo | None
) -> str | None:
    if info is None or info.file_size > _MAX_CONTENT_TYPES_BYTES:
        return None
    try:
        haystack = archive.read(info).lower()
    except (zipfile.BadZipFile, OSError, RuntimeError, zlib.error):
        return None
    for marker, package_mime in _CONTENT_TYPE_MARKERS:
        if marker in haystack:
            return package_mime
    return None


def _mime_from_odf_member(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo | None
) -> str | None:
    if info is None or info.file_size > _MAX_PACKAGE_MIMETYPE_BYTES:
        return None
    try:
        declared = archive.read(info).strip()
    except (zipfile.BadZipFile, OSError, RuntimeError, zlib.error):
        return None
    if not (
        declared.startswith(b"application/vnd.oasis.opendocument.")
        or declared == b"application/epub+zip"
    ):
        return None
    try:
        return declared.decode("ascii")
    except UnicodeDecodeError:
        return None


def _detect_with_magic_or_sniff(content: bytes) -> str:
    if not is_magic_available():
        return _sniff(content[:8192]) or _DEFAULT_MIME_TYPE
    import magic  # ty:ignore[unresolved-import]

    try:
        detected = magic.Magic(mime=True).from_buffer(content)
    except Exception:
        return _DEFAULT_MIME_TYPE
    if not isinstance(detected, str) or not detected:
        return _DEFAULT_MIME_TYPE
    return detected


def detect_mime_from_buffer(content: bytes) -> str:
    """Sniff the MIME type of ``content``; octet-stream when unknown."""
    return refine_detected_mime(_detect_with_magic_or_sniff(content), content)


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
            detected = magic.Magic(mime=True).from_file(str(path))
        except Exception:
            return None
        if not isinstance(detected, str) or not detected:
            return None
        return refine_detected_mime(detected, path)
    try:
        with path.open("rb") as fh:
            detected = _sniff(fh.read(8192))
    except OSError:
        return None
    if detected is None:
        return None
    return refine_detected_mime(detected, path)
