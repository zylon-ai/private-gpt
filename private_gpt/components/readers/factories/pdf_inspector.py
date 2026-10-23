from private_gpt.components.readers.base_reader import IngestionReader
from private_gpt.components.readers.factories.base import ReaderFactory
from private_gpt.utils.dependencies import format_missing_dependency_message


class PdfInspectorReaderFactory(ReaderFactory):
    def create_reader(self, extension: str | None = None) -> IngestionReader:
        del extension

        try:
            from private_gpt.components.readers.pdf_inspector.pdf_inspector_reader import (
                PdfInspectorReader,
            )
        except ImportError as e:
            raise ImportError(
                format_missing_dependency_message(
                    "PDF Inspector reader",
                    extras="ingest-pdf-inspector",
                )
            ) from e

        return PdfInspectorReader()
