import logging
from typing import Any

from llama_index.core.schema import BaseNode

from private_gpt.components.ingest.metadata_helper import MetadataHelper
from private_gpt.components.ingest.progress.errors import IngestionValidationErrors
from private_gpt.components.ingest.utils import FileInfo, should_ignore_mime_mismatch
from private_gpt.settings.settings import settings
from private_gpt.utils.mime import is_magic_available

logger = logging.getLogger(__name__)


class IngestionHelper:
    @staticmethod
    def validate_file_info(
        file_info: FileInfo,
    ) -> tuple[list[str], list[str]]:
        if not file_info:
            raise RuntimeError("Failed to extract file info")

        errors = []
        warnings = []

        # Check if the file info has size
        if file_info.file_size is None or file_info.file_size <= 0:
            errors.append(IngestionValidationErrors.INVALID_FILE_SIZE)

        # Check if the file info has extension
        if not file_info.extension:
            errors.append(IngestionValidationErrors.UNKNOWN_FILE_EXTENSION)

        if is_magic_available() and not file_info.actual_mime_type:
            errors.append(IngestionValidationErrors.UNKNOWN_FILE_EXTENSION)

        if (
            file_info.guest_mime_type
            and file_info.actual_mime_type
            and file_info.guest_mime_type != file_info.actual_mime_type
            and not should_ignore_mime_mismatch(
                file_info.guest_mime_type, file_info.actual_mime_type
            )
        ):
            logger.info(
                "MIME type mismatch: guest_mime_type=%s, actual_mime_type=%s",
                file_info.guest_mime_type,
                file_info.actual_mime_type,
            )
            errors.append(IngestionValidationErrors.MISMATCHED_MIME_TYPE)

        # Check if the file is too large
        if (
            file_info.file_size
            and file_info.file_size > settings().data.limits.max_file_size
        ):
            warnings.append(IngestionValidationErrors.BIG_FILE_SIZE)

        # Check if there is any error
        if "error" in file_info.config:
            errors.append(IngestionValidationErrors.MALFORMED_FILE)

        # Check if the file has too many pages
        try:
            pages = int(file_info.config.get("pages", 0))
            if pages > 0 and pages > settings().data.limits.max_file_pages:
                warnings.append(IngestionValidationErrors.BIG_FILE_PAGES)
        except (ValueError, TypeError):
            pass  # pages is not convertible to int, skip validation

        # Check if the file is special
        # TODO: Disabled for now. Please re-enable
        # test_validate_file_info_real_pdf_with_forms,
        # test_validate_file_info_real_pdf_with_annotations
        # and test_validate_file_info_pdf_with_forms

        # special_file = file_info.config.get("special", False)
        # if special_file:
        #     warnings.append(IngestionValidationErrors.SPECIAL_FILE)

        # Check if the file is encrypted.
        # TODO: Give support to encrypted files
        is_encrypted = file_info.config.get("is_encrypted", False)
        if is_encrypted:
            warnings.append(IngestionValidationErrors.SPECIAL_ENCRYPTED_FILE)

        return list(set(errors)), list(set(warnings))

    @staticmethod
    def exclude_metadata(
        nodes: list[BaseNode], file_metadata: dict[str, Any] | None = None
    ) -> None:
        logger.debug("Excluding metadata from count=%s nodes", len(nodes))
        for node in nodes:
            # We don't want the LLM to receive these metadata in the context
            MetadataHelper.exclude_metadata(node)

            # Add all keys of the file_metadata to excluded_llm_metadata_keys
            if file_metadata:
                for key in file_metadata:
                    MetadataHelper.exclude_key_metadata(
                        node=node,
                        key=key,
                        exclude_llm=True,
                        exclude_from_embed=False,
                    )
