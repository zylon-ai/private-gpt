import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from injector import inject, singleton
from llama_index.core.schema import BaseNode
from pydantic import BaseModel

from private_gpt.artifact_index.artifact_exception import InvalidFileError
from private_gpt.celery.notify import NotifyProtocol, ProgressStatus, notify_progress
from private_gpt.components.ingest.fake_progress import (
    calculate_validation_timing,
)
from private_gpt.components.ingest.ingest_helper import IngestionHelper
from private_gpt.components.ingest.progress.errors import (
    IngestionLoadErrors,
    IngestionParseErrors,
)
from private_gpt.components.ingest.progress.models import ValidationProgressStatus
from private_gpt.components.ingest.utils import (
    FileInfo,
    convert_unsupported_file,
    convert_unsupported_file_as_fallback,
    get_file_info,
    get_file_name,
    get_filesize,
)
from private_gpt.components.readers.reader_component import ReaderComponent
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FileParseResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    nodes: list[BaseNode]
    reader: str


@singleton
class ParseComponent:
    @inject
    def __init__(
        self,
        settings: Settings,
        reader_component: ReaderComponent,
    ) -> None:
        self.reader_component = reader_component
        self._generate_fake_percentage = settings.data.enable_fake_progress

    def load_and_validate_file(
        self,
        file_data: Path,
        file_metadata: dict[str, Any] | None = None,
        notify: Callable[[ProgressStatus], None] = lambda x: None,
        warnings: list[str] | None = None,
    ) -> tuple[FileInfo, list[str], list[str]]:
        file_size = get_filesize(file_data)
        interval, jitter = calculate_validation_timing(file_size=file_size)

        with notify_progress(
            notify=notify,
            status_class=ValidationProgressStatus,
            warnings=warnings,
            generate_fake_percentage=self._generate_fake_percentage,
            generate_fake_percentage_interval_ms=int(interval * 1000)
            if interval
            else None,
            generate_fake_percentage_jitter=jitter,
        ) as progress:
            logger.info("Validating file: %s", file_data)
            file_info = self._get_file_info(file_data, file_metadata, progress)
            errors, warnings = self._validate_file(file_info, progress)
            logger.info("Finished validating file: %s", file_data)
            return file_info, errors, warnings

    def file_to_nodes(
        self,
        file_info: FileInfo,
        file_metadata: dict[str, Any] | None = None,
        reader_name: str | None = None,
        notification: NotifyProtocol | None = None,
        warnings: list[str] | None = None,
    ) -> FileParseResult:
        converted_file = convert_unsupported_file(file_info)
        resolved_reader = reader_name or self._resolve_reader(converted_file.extension)

        try:
            nodes = self._load_data(
                converted_file,
                file_metadata,
                notification=notification,
                warnings=warnings,
                reader_name=resolved_reader,
            )
        except RuntimeError as e:
            raise InvalidFileError(errors=[IngestionParseErrors.PARSING_FAILURE]) from e
        except Exception as e:
            logger.error("Error loading file: %s", e, exc_info=True)
            converted_fallback = convert_unsupported_file_as_fallback(file_info)
            if converted_fallback:
                if notification:
                    notification(
                        percentage=0,
                        warnings=[IngestionParseErrors.FALLBACK_TO_PDF_TO_TEXT],
                    )
                nodes = self._load_data(
                    converted_fallback,
                    file_metadata,
                    notification=notification,
                    warnings=warnings,
                    reader_name=resolved_reader,
                )
            else:
                nodes = []

        if not nodes:
            logger.info("No valid nodes found in the file.")
            raise InvalidFileError(
                errors=[IngestionLoadErrors.NO_VALID_FILES], warnings=warnings
            )

        return FileParseResult(nodes=nodes, reader=resolved_reader)

    def _resolve_reader(self, extension: str | None) -> str:
        names = self.reader_component.get_reader_names(extension=extension or "")
        return names[0] if names else "text"

    def _get_file_info(
        self,
        file_data: Path,
        file_metadata: dict[str, Any] | None,
        progress: NotifyProtocol | None = None,
    ) -> FileInfo:
        file_name = get_file_name(file_metadata)
        return get_file_info(file_data, file_name=file_name, progress=progress)

    def _validate_file(
        self,
        file_info: FileInfo,
        progress: NotifyProtocol,
    ) -> tuple[list[str], list[str]]:
        errors, warnings = IngestionHelper.validate_file_info(file_info)
        if errors:
            logger.info("Validation errors: %s", errors)
            raise InvalidFileError(errors=errors, warnings=warnings)
        if warnings:
            logger.info("Validation warnings: %s", warnings)
            progress(percentage=100, warnings=warnings)
        return errors, warnings

    def _load_data(
        self,
        file_info: FileInfo,
        file_metadata: dict[str, Any] | None,
        notification: NotifyProtocol | None = None,
        warnings: list[str] | None = None,
        reader_name: str | None = None,
    ) -> list[BaseNode]:
        return asyncio.run(
            self._aload_data(
                file_info=file_info,
                file_metadata=file_metadata,
                notification=notification,
                warnings=warnings,
                reader_name=reader_name,
            )
        )

    async def _aload_data(
        self,
        file_info: FileInfo,
        file_metadata: dict[str, Any] | None,
        notification: NotifyProtocol | None = None,
        warnings: list[str] | None = None,
        reader_name: str | None = None,
    ) -> list[BaseNode]:
        if reader_name:
            loader = self.reader_component.get_reader(reader_name, file_info.extension)
        else:
            loader = self.reader_component.get_reader_by_extension(
                file_info.extension or ""
            )
        nodes: list[BaseNode] = []
        async for node in loader.lazy_load_data(
            file_info,
            extra_info=file_metadata,
            notification=notification,
            warnings=warnings,
        ):
            nodes.append(node)
        return nodes
