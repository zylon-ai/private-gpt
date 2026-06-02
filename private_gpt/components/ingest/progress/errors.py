import enum


class IngestionValidationErrors(enum.StrEnum):
    # Invalid files
    INVALID_FILE_SIZE = "zpgt.ingest.invalid_file_size.error"
    MALFORMED_FILE = "zpgt.ingest.malformed_file.error"

    # Unknown information
    UNKNOWN_FILE_EXTENSION = "zpgt.ingest.invalid_file_extension.error"
    MISMATCHED_MIME_TYPE = "zpgt.ingest.mismatched_mime_type.error"

    # Files are too large
    BIG_FILE_SIZE = "zpgt.ingest.big_file_size.error"
    BIG_FILE_PAGES = "zpgt.ingest.big_file_pages.error"

    # Special docs
    SPECIAL_FILE = "zpgt.ingest.special_file.error"
    SPECIAL_ENCRYPTED_FILE = "zpgt.ingest.encrypted_file.error"


class IngestionParseErrors(enum.StrEnum):

    # Errors during the parsing process
    PARSING_FAILURE = "zpgt.ingest.parsing_failure.error"

    # Errors during the partitioning process
    MOVING_TO_OCR = "zpgt.ingest.moving_to_ocr.error"

    # Warnings when we fallback to PDF to text extraction
    FALLBACK_TO_PDF_TO_TEXT = "zpgt.ingest.fallback_to_pdf_to_text.warning"
    USING_VLM_FOR_EXTRACTION = "zpgt.ingest.using_vlm_for_extraction.warning"


class IngestionLoadErrors(enum.StrEnum):

    # Errors during the loading process
    NO_VALID_FILES = "zpgt.ingest.no_valid_files.error"
    NO_VALID_NODES = "zpgt.ingest.no_valid_nodes.error"

    # Errors during the retrieving process
    EXCEEDS_MAX_NODES = "zpgt.ingest.exceeds_max_nodes.error"
