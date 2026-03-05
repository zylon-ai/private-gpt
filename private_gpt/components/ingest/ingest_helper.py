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

    #Modification to provide better support for file ingest with charmap issues
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
            # Read as a plain text
            string_reader = StringIterableReader()
            return string_reader.load_data([IngestionHelper._read_all_text(file_data)])
    
        logger.debug("Specific reader found for extension=%s", extension)
        documents = reader_cls().load_data(file_data)
        
        return documents

    #new method being called from _load_file_to_documents
    @staticmethod
    def _read_all_text(file_data: Path) -> str:
        try:
            # Read raw bytes first
            raw_bytes = file_data.read_bytes()
            
            # Use chardet to detect encoding
            detected = chardet.detect(raw_bytes)
            encoding = detected["encoding"] or 'utf-8'  # Fallback to utf-8 if detection fails
            confidence = detected.get('confidence', 0)
            
            logger.debug(f"Detected encoding {encoding} with confidence {confidence} for {file_data}")
            
            # Try the detected encoding first
            try:
                text = raw_bytes.decode(encoding)
            except UnicodeDecodeError:
                # If detected encoding fails, try common encodings
                for fallback_encoding in ['utf-8', 'cp1252', 'iso-8859-1', 'latin1']:
                    try:
                        text = raw_bytes.decode(fallback_encoding)
                        logger.debug(f"Successfully decoded with fallback encoding: {fallback_encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # If all encodings fail, use 'replace' error handler with utf-8
                    text = raw_bytes.decode('utf-8', errors='replace')
                    logger.warning(f"Falling back to UTF-8 with replacement for {file_data}")
            
            # Clean up the text
            cleaned_text = (text.encode('utf-8', errors='replace')
                        .decode('utf-8')
                        .replace('\udc58', '')  # Remove specific problematic Unicode chars
                        .replace('\x00', '')     # Remove null bytes
                        .replace('\ufffd', '')   # Remove replacement character
                        .strip())                # Remove leading/trailing whitespace
            
            if not cleaned_text:
                logger.warning(f"Cleaned text is empty for {file_data}")
                
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error processing file {file_data}: {str(e)}")
            raise

    @staticmethod
    def _exclude_metadata(documents: list[Document]) -> None:
        logger.debug("Excluding metadata from count=%s documents", len(documents))
        for document in documents:
            document.metadata["doc_id"] = document.doc_id
            # We don't want the Embeddings search to receive this metadata
            document.excluded_embed_metadata_keys = ["doc_id"]
            # We don't want the LLM to receive these metadata in the context
            document.excluded_llm_metadata_keys = ["file_name", "doc_id", "page_label"]
