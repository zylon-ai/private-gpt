import logging
from pathlib import Path

from llama_index.core.readers import StringIterableReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.json import JSONReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


# Inspired by the `llama_index.core.readers.file.base` module
def _try_loading_included_file_formats() -> dict[str, type[BaseReader]]:
    try:
        from llama_index.readers.file.docs import (  # type: ignore
            DocxReader,
            HWPReader,
            PDFReader,
        )
        from llama_index.readers.file.epub import EpubReader  # type: ignore
        from llama_index.readers.file.image import ImageReader  # type: ignore
        from llama_index.readers.file.ipynb import IPYNBReader  # type: ignore
        from llama_index.readers.file.markdown import MarkdownReader  # type: ignore
        from llama_index.readers.file.mbox import MboxReader  # type: ignore
        from llama_index.readers.file.slides import PptxReader  # type: ignore
        from llama_index.readers.file.tabular import PandasCSVReader  # type: ignore
        from llama_index.readers.file.video_audio import (  # type: ignore
            VideoAudioReader,
        )
        from llama_index.readers.file.xml import XMLReader  # type: ignore
        from private_gpt.components.ingest.readers.xlsx_parser import XLSXParser  # type: ignore
        from private_gpt.components.ingest.readers.html_parser import HTMLParser  # type: ignore
        from private_gpt.components.ingest.readers.doc_parser import DOCParser  # type: ignore
    except ImportError as e:
        raise ImportError("`llama-index-readers-file` package not found") from e

    default_file_reader_cls: dict[str, type[BaseReader]] = {
        ".hwp": HWPReader,
        ".pdf": PDFReader,
        ".docx": DocxReader,
        ".pptx": PptxReader,
        ".ppt": PptxReader,
        ".pptm": PptxReader,
        ".jpg": ImageReader,
        ".png": ImageReader,
        ".jpeg": ImageReader,
        ".mp3": VideoAudioReader,
        ".mp4": VideoAudioReader,
        ".csv": PandasCSVReader,
        ".epub": EpubReader,
        ".md": MarkdownReader,
        ".mbox": MboxReader,
        ".ipynb": IPYNBReader,
        # // delete desktop.ini and *.eps files. 
        ".markdown": MarkdownReader,
        ".html": HTMLParser,
        ".htm": HTMLParser,
        ".xlsx": XLSXParser,
        ".xml": XMLReader,
        ".eps": ImageReader,
        ".tif": ImageReader,
        ".gif": ImageReader,
        ".doc": DOCParser,
    }
    return default_file_reader_cls


# Patching the default file reader to support other file types
FILE_READER_CLS = _try_loading_included_file_formats()
FILE_READER_CLS.update(
    {
        ".json": JSONReader,
    }
)


class IngestionHelper:
    """Helper class to transform a file into a list of documents.

    This class should be used to transform a file into a list of documents.
    These methods are thread-safe (and multiprocessing-safe).
    """

    @staticmethod
    def transform_file_into_documents(
        file_name: str, file_data: Path
    ) -> list[Document]:
        documents = IngestionHelper._load_file_to_documents(file_name, file_data)
        for document in documents:
            document.metadata["file_name"] = file_name
        IngestionHelper._exclude_metadata(documents)
        return documents

    @staticmethod
    def _load_file_to_documents(file_name: str, file_data: Path) -> list[Document]:
        logger.debug("Transforming file_name=%s into documents", file_name)
        extension = Path(file_name).suffix
        reader_cls = FILE_READER_CLS.get(extension)
        if reader_cls is None:
            logger.debug(
                "No reader found for extension=%s, using default string reader",
                extension,
            )
            # Attempt to read as a plain text with UTF-8 encoding, if fails read as ASCII
            try:
                file_content = file_data.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    logger.debug(
                        "Failed to read file_name=%s as UTF-8 encoded text, trying with ASCII",
                        file_name,
                    )
                    # Fallback to ASCII decoding
                    file_content = file_data.read_text(encoding='ascii')
                except UnicodeDecodeError:
                    logger.debug(
                        "Failed to read file_name=%s as ASCII encoded text, returning empty document",
                        file_name,
                    )
                    return []
            string_reader = StringIterableReader()
            return string_reader.load_data([file_content])

        logger.debug("Specific reader found for extension=%s", extension)
        try:
            return reader_cls().load_data(file_data)
        except Exception as e:
            logger.debug(
                "Failed to read file_name=%s e=%s",
                file_name,
                e
            )
            return []

    @staticmethod
    def _exclude_metadata(documents: list[Document]) -> None:
        logger.debug("Excluding metadata from count=%s documents", len(documents))
        for document in documents:
            document.metadata["doc_id"] = document.doc_id
            # We don't want the Embeddings search to receive this metadata
            document.excluded_embed_metadata_keys = ["doc_id"]
            # We don't want the LLM to receive these metadata in the context
            document.excluded_llm_metadata_keys = ["file_name", "doc_id", "page_label"]
