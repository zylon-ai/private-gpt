import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from private_gpt.celery.notify import NotifyProtocol
from private_gpt.components.ingest.metadata_helper import MetadataKeys


class FileInfo(BaseModel):
    file_name: str | None = None
    extension: str | None = None
    file_data: Path
    guest_mime_type: str | None = None
    actual_mime_type: str | None = None
    file_size: int | None = None
    encoding: str | None = None
    hash: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


def exist_file(file_data: Path) -> bool:
    """Check if the file exists."""
    return file_data.exists()


def get_filename(file_data: Path) -> str:
    """Extract the file name from the path."""
    import os

    return os.path.basename(file_data)


def get_filesize(file_data: Path) -> int | None:
    """Get the file size in bytes."""
    try:
        import os

        return os.path.getsize(file_data)
    except Exception:
        return None


def get_extension(file_name: str) -> str | None:
    """Extract the file extension from the path."""
    import os

    file = os.path.splitext(file_name)
    return file[1].lower() if len(file) > 1 and file[1] else None


def get_guest_mime_type(file_data: Path) -> str | None:
    """Get the MIME type based on the file extension."""
    try:
        import mimetypes

        mime_type, _ = mimetypes.guess_type(file_data)
        return mime_type
    except Exception:
        return None


def get_actual_mime_type(file_data: Path) -> str | None:
    try:
        import magic

        mime_detector = magic.Magic(mime=True)
        return mime_detector.from_file(file_data)
    except ImportError:
        return None
    except Exception:
        return None


def should_ignore_mime_mismatch(guest_mime: str, actual_mime: str) -> bool:
    """Determine if a MIME type mismatch should be ignored.

    Based on known valid combinations where both types represent the same content.
    """
    # Define valid MIME type pairs that represent the same content
    valid_pairs = {
        # Text and markup formats
        frozenset({"text/html", "text/plain"}),
        frozenset({"text/markdown", "text/plain"}),
        frozenset({"application/xml", "text/plain"}),
        frozenset({"text/xml", "text/plain"}),
        frozenset({"application/json", "text/plain"}),
        frozenset({"application/javascript", "text/plain"}),
        frozenset({"application/x-javascript", "text/plain"}),
        frozenset({"message/rfc822", "text/plain"}),
        frozenset({"application/x-appleworks3", "text/plain"}),
        # HTML can contains other types inline, making very difficult to detect
        frozenset({"text/html", "application/javascript"}),
        frozenset({"text/html", "application/x-javascript"}),
        frozenset({"text/html", "application/json"}),
        frozenset({"text/html", "text/xml"}),
        frozenset({"text/html", "application/xml"}),
        frozenset({"text/html", "text/css"}),
        # Microsoft Word documents
        frozenset(
            {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword",
            }
        ),
        frozenset(
            {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/octet-stream",
            }
        ),
        frozenset(
            {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/zip",
            }
        ),
        frozenset({"application/msword", "application/zip"}),
        # Microsoft Excel documents
        frozenset(
            {
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.ms-excel",
            }
        ),
        frozenset(
            {
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/octet-stream",
            }
        ),
        frozenset(
            {
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/zip",
            }
        ),
        frozenset({"application/vnd.ms-excel", "application/octet-stream"}),
        frozenset({"application/vnd.ms-excel", "application/zip"}),
        # Microsoft PowerPoint documents
        frozenset(
            {
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "application/vnd.ms-powerpoint",
            }
        ),
        frozenset(
            {
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "application/octet-stream",
            }
        ),
        frozenset(
            {
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "application/zip",
            }
        ),
        frozenset({"application/vnd.ms-powerpoint", "application/zip"}),
        # PDF documents
        frozenset({"application/pdf", "application/octet-stream"}),
        # OpenDocument formats
        frozenset(
            {"application/vnd.oasis.opendocument.text", "application/octet-stream"}
        ),
        frozenset({"application/vnd.oasis.opendocument.text", "application/zip"}),
        frozenset(
            {
                "application/vnd.oasis.opendocument.spreadsheet",
                "application/octet-stream",
            }
        ),
        frozenset(
            {"application/vnd.oasis.opendocument.spreadsheet", "application/zip"}
        ),
        frozenset(
            {
                "application/vnd.oasis.opendocument.presentation",
                "application/octet-stream",
            }
        ),
        frozenset(
            {"application/vnd.oasis.opendocument.presentation", "application/zip"}
        ),
        # Common archive formats
        frozenset({"application/zip", "application/octet-stream"}),
        frozenset({"application/x-rar-compressed", "application/octet-stream"}),
        frozenset({"application/x-7z-compressed", "application/octet-stream"}),
        frozenset({"application/gzip", "application/octet-stream"}),
        # Image formats
        frozenset({"image/jpeg", "application/octet-stream"}),
        frozenset({"image/png", "application/octet-stream"}),
        frozenset({"image/gif", "application/octet-stream"}),
        frozenset({"image/webp", "application/octet-stream"}),
        frozenset({"image/svg+xml", "text/plain"}),
        frozenset({"image/svg+xml", "application/xml"}),
        # Audio/Video formats
        frozenset({"audio/mpeg", "application/octet-stream"}),
        frozenset({"video/mp4", "application/octet-stream"}),
        frozenset({"audio/wav", "application/octet-stream"}),
        # CSV files
        frozenset({"text/csv", "text/plain"}),
        frozenset({"text/tsv", "text/plain"}),
    }

    current_pair = frozenset({guest_mime, actual_mime})
    return current_pair in valid_pairs


def detect_encoding(file_data: Path) -> str | None:
    try:
        import chardet.universaldetector

        detector = chardet.universaldetector.UniversalDetector()
        initial_chunk_size = 4096  # Start with 4KB
        max_multiplier = 8  # How many times we'll increase the chunk size

        with open(file_data, "rb") as file:
            chunk_size = initial_chunk_size
            position = 0

            for _ in range(max_multiplier):
                file.seek(position)
                chunk = file.read(chunk_size)

                if not chunk:  # End of file reached
                    break

                detector.feed(chunk)

                if detector.done:
                    break

                # Increase sample size for next iteration if needed
                position = 0  # Reset to start for larger chunk
                chunk_size *= 2  # Double the chunk size

            # If still not confident, read the entire file
            if not detector.done:
                file.seek(0)
                chunk = file.read()
                detector.feed(chunk)

        detector.close()
        confidence: float = detector.result["confidence"]
        # Files with little text and utf-8 emoji are detected
        # as Windows-1252 with low confidence
        return detector.result["encoding"] if confidence >= 0.9 else None

    except Exception:
        return None


def calculate_file_hash(file_data: Path) -> str | None:
    """Calculate the hash of a file."""
    try:
        import hashlib

        # Open the file and calculate the hash
        with open(file_data, "rb") as f:
            hash_value = hashlib.sha256()
            for chunk in iter(lambda: f.read(4096), b""):
                hash_value.update(chunk)

        return hash_value.hexdigest()
    except Exception:
        return None


def extract_pdf_info(file_data: Path) -> dict[str, Any | None]:
    """Extract specific information from PDF files."""
    config: dict[str, Any | None] = {}

    try:
        from pypdf import PdfReader

        with open(file_data, "rb") as f:
            reader = PdfReader(f)

            def extract_num_pages() -> int | None:
                """Extract the number of pages from the PDF."""
                try:
                    return len(reader.pages)
                except Exception:
                    return None

            def extract_encryption_status() -> bool | None:
                """Check if the PDF is encrypted."""
                try:
                    return reader.is_encrypted
                except Exception:
                    return None

            def has_forms() -> bool | None:
                """Check if the PDF contains forms (AcroForm)."""
                try:
                    return (
                        bool(reader.trailer.get("/AcroForm"))
                        or len(reader.get_form_text_fields()) > 0
                    )
                except Exception:
                    return None

            def has_images() -> bool | None:
                """Check if the PDF contains images."""
                try:
                    return any(page.images for page in reader.pages)
                except Exception:
                    return None

            def has_annotations() -> bool | None:
                """Check if the PDF contains annotations."""
                try:
                    return any(page.get("/Annots") for page in reader.pages)
                except Exception:
                    return None

            def has_attachments() -> bool | None:
                """Check if the PDF contains attachments."""
                try:
                    return len(reader.attachments.items()) > 0
                except Exception:
                    return None

            # Extract pdf information
            config["pages"] = extract_num_pages()
            config["is_encrypted"] = extract_encryption_status()
            config["has_images"] = has_images()

            # Check for special PDFs
            config["has_forms"] = has_forms()
            config["has_annotations"] = has_annotations()
            config["has_attachments"] = has_attachments()
            config["special"] = any(
                [
                    config["has_forms"],
                    config["has_annotations"],
                    config["has_attachments"],
                ]
            )
    except Exception as e:
        config["error"] = str(e)

    config = {k: v for k, v in config.items() if v is not None}
    return config


def extract_config(file_data: Path, extension: str | None) -> dict[str, int | None]:
    """Extract specific config based on the file type."""
    match extension:
        case ".pdf":
            return extract_pdf_info(file_data)
        case _:
            return {}


def get_file_name(
    file_metadata: dict[str, Any] | None,
) -> str | None:
    # Extracting the file name to help detect the file type through the extension
    file_name: str | None = (
        file_metadata.get(MetadataKeys.FILENAME.value) if file_metadata else None
    )
    # In case the file name does not contain an extension, we discard it
    if file_name and len(Path(file_name).suffix) == 0:
        file_name = None

    return file_name


def get_file_info(
    file_data: Path, file_name: str | None, progress: NotifyProtocol | None = None
) -> FileInfo:
    """Function to extract file information."""
    if not exist_file(file_data):
        raise FileNotFoundError(f"File not found: {file_data}")

    steps = 7
    current_step = 0

    def notify() -> None:
        if progress is None:
            return
        nonlocal current_step
        current_step += 1
        if current_step <= steps:
            progress(percentage=current_step * 100 // steps)

    file_name = file_name
    extension = get_extension(file_name) if file_name else None
    notify()

    file_size = get_filesize(file_data)
    notify()

    guest_mime_type = get_guest_mime_type(file_data)
    notify()

    actual_mime_type = get_actual_mime_type(file_data)
    notify()

    encoding = detect_encoding(file_data)
    notify()

    hash = calculate_file_hash(file_data)
    notify()

    config = extract_config(file_data, extension)
    notify()

    return FileInfo(
        file_name=file_name,
        file_size=file_size,
        file_data=file_data,
        extension=extension,
        guest_mime_type=guest_mime_type,
        actual_mime_type=actual_mime_type,
        encoding=encoding,
        hash=hash,
        config=config,
    )


def convert_file_with_libreoffice(
    file_info: FileInfo, target_extension: str, raise_in_exception: bool = False
) -> Path:
    output_path = file_info.file_data.with_suffix(target_extension)
    try:
        subprocess.run(
            [
                "soffice",
                "--headless",
                "--convert-to",
                target_extension.lstrip("."),
                "--outdir",
                str(output_path.parent),
                str(file_info.file_data),
            ],
            check=True,
        )
    except FileNotFoundError as e:
        raise ImportError(
            "LibreOffice is required for file conversion. Please install LibreOffice."
        ) from e
    except subprocess.CalledProcessError as e:
        if raise_in_exception:
            raise RuntimeError(
                f"Failed to convert {file_info.file_data} to {target_extension}: {e}"
            ) from e
        return file_info.file_data  # Return the original file on failure
    return output_path


def convert_file(
    file_info: FileInfo,
    conversion_func: Callable[[FileInfo, str, bool], Path],
    target_extension: str,
    raise_in_exception: bool = False,
) -> FileInfo:
    try:
        converted_path = conversion_func(
            file_info, target_extension, raise_in_exception
        )
        new_file_name = (
            file_info.file_name.replace(file_info.extension, converted_path.suffix)
            if file_info.file_name and file_info.extension
            else None
        )
        return get_file_info(converted_path, file_name=new_file_name)
    except Exception as e:
        if raise_in_exception:
            raise RuntimeError(f"Conversion failed: {e}") from e
        return file_info  # Return the original FileInfo on failure


def convert_unsupported_file(
    file_info: FileInfo, raise_in_exception: bool = False
) -> FileInfo:
    extensions_map = {
        ".xls": (".xlsx", convert_file_with_libreoffice),
        ".doc": (".docx", convert_file_with_libreoffice),
        ".ppt": (".pptx", convert_file_with_libreoffice),
    }
    if file_info.extension not in extensions_map:
        return file_info  # Return original FileInfo if no conversion is available

    target_extension, conversion_func = extensions_map[file_info.extension]
    return convert_file(
        file_info, conversion_func, target_extension, raise_in_exception
    )


def convert_unsupported_file_as_fallback(
    file_info: FileInfo, raise_in_exception: bool = False
) -> FileInfo | None:
    """Convert unsupported files as a fallback.

    Since the conversion is not guaranteed to work with Docling,
    we try to convert into PDF format as a fallback.
    """
    extensions_map = {
        ".doc": (".pdf", convert_file_with_libreoffice),
        ".ppt": (".pdf", convert_file_with_libreoffice),
        ".docx": (".pdf", convert_file_with_libreoffice),
        ".pptx": (".pdf", convert_file_with_libreoffice),
    }
    if file_info.extension not in extensions_map:
        return file_info  # Return original FileInfo if no conversion is available

    target_extension, conversion_func = extensions_map[file_info.extension]
    return convert_file(
        file_info, conversion_func, target_extension, raise_in_exception
    )
