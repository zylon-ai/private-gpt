import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from llama_index.core.schema import BaseNode, Document

from private_gpt.components.readers.text.text_reader import TextReader

logger = logging.getLogger(__name__)

# PDF types that pdf-inspector itself considers to require OCR outright.
_OCR_REQUIRED_TYPES = {"scanned", "image_based"}

# Minimum classification confidence to trust a "mixed" PDF that reports no
# pages needing OCR. Below this we don't trust the classification and defer
# to the existing OCR-capable reader instead of risking incomplete text.
_MIN_MIXED_CONFIDENCE = 0.6


class PdfInspectorFallbackError(Exception):
    """Raised when pdf-inspector cannot confidently extract a PDF without OCR.

    Signals the caller to fall back to the next reader in the configured
    chain (the existing OCR-capable flow), rather than being handled here.
    """


class PdfInspectorReader(TextReader):
    def lazy_document_load(
        self,
        file_path: Path,
        encoding: str | None = None,
        extra_info: dict[str, Any] | None = None,
    ) -> Iterator[BaseNode]:
        del encoding

        try:
            import pdf_inspector  # type: ignore[import-not-found]

            logger.debug("Starting pdf-inspector classification of file: %s", file_path)
            result = pdf_inspector.process_pdf(str(file_path))
        except Exception as e:
            logger.warning(
                "pr failed to process '%s'; falling back to OCR reader: %s",
                file_path,
                e,
            )
            raise PdfInspectorFallbackError(
                f"pdf-inspector failed to process '{file_path}'."
            ) from e

        if result.pages_needing_ocr:
            logger.info(
                "pdf-inspector detected pages needing OCR in '%s'; "
                "falling back to OCR reader.",
                file_path,
            )
            raise PdfInspectorFallbackError(
                f"'{file_path}' has pages requiring OCR: {result.pages_needing_ocr}."
            )

        if result.pdf_type in _OCR_REQUIRED_TYPES:
            logger.info(
                "pdf-inspector classified '%s' as '%s'; falling back to OCR reader.",
                file_path,
                result.pdf_type,
            )
            raise PdfInspectorFallbackError(
                f"'{file_path}' classified as '{result.pdf_type}' requires OCR."
            )

        if result.pdf_type == "mixed" and result.confidence < _MIN_MIXED_CONFIDENCE:
            logger.info(
                "pdf-inspector classified '%s' as 'mixed' with low confidence "
                "(%.2f); falling back to OCR reader.",
                file_path,
                result.confidence,
            )
            raise PdfInspectorFallbackError(
                f"'{file_path}' classified as 'mixed' with low confidence "
                f"({result.confidence:.2f})."
            )

        logger.debug("Finished pdf-inspector extraction of file: %s", file_path)

        yield Document(
            text=result.markdown or "",
            extra_info=extra_info if extra_info is not None else {},
        )
