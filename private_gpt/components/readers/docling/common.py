from typing import Any

from private_gpt.components.readers.docling.utils import (
    convert_to_easyocr_lang,
    convert_to_ocrmac_lang,
    convert_to_rapidocr_lang,
    convert_to_tesseract_lang,
)
from private_gpt.components.readers.nodes.image_node import IMAGE_PLACEHOLDER
from private_gpt.settings.settings import settings

# Constants
DEFAULT_IMAGE_PLACEHOLDER = "<!-- image -->"
EMBEDDED_IMAGES = settings().docling.image_mode == "embedded"
IMAGE_DUMMY = IMAGE_PLACEHOLDER
IMAGE_RESOLUTION_SCALE = 2.0

PAGE_PLACEHOLDER = "\n<!-- page -->\n"


def get_ocr_langs() -> list[str] | None:
    """Get the OCR languages.

    Returns:
        list[str]: List of OCR languages.
    """
    langs: list[str] | None = settings().docling.langs
    if not langs:
        raise ValueError("No OCR languages specified.")
    match settings().docling.ocr_model:
        case "easyocr":
            langs = [convert_to_easyocr_lang(lang) for lang in langs]
        case "tesseract":
            langs = [convert_to_tesseract_lang(lang) for lang in langs]
        case "rapidocr":
            langs = [convert_to_rapidocr_lang(lang) for lang in langs]
        case "ocrmac":
            langs = [convert_to_ocrmac_lang(lang) for lang in langs]
        case _:
            raise ValueError(f"OCR model {settings().docling.ocr_model} not supported")
    return langs


async def calculate_file_priority(
    file_bytes: bytes, pages: int | None = None, **kwargs: Any
) -> int:
    """Calculate processing priority based on file size and page count.

    Priority levels:
    - 0: High priority (small files < 1MB and <= 100 pages)
    - 1: Low priority (files > 10MB or > 50 pages)
    """
    file_size = len(file_bytes)

    # High priority: files under 1MB
    if file_size < 1_000_000 and (pages is None or pages <= 100):
        return 0

    # Low priority: files over 10MB or more than 50 pages
    return 1
