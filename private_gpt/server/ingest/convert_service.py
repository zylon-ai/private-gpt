import logging
import os
import tempfile
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, AnyStr, BinaryIO

from injector import inject, singleton

from private_gpt.components.ingest.parse_component import (
    FileParseResult,
    ParseComponent,
)

logger = logging.getLogger(__name__)


@singleton
class ConvertService:
    @inject
    def __init__(self, parse_component: ParseComponent) -> None:
        self.parse_component = parse_component

    def convert_file(
        self,
        file_data: Path,
        file_metadata: dict[str, Any] | None = None,
        reader: str | None = None,
    ) -> FileParseResult:
        file_info, _, _ = self.parse_component.load_and_validate_file(
            file_data, file_metadata
        )
        return self.parse_component.file_to_nodes(file_info, file_metadata, reader)

    def data_path_from_data(
        self,
        file_data: AnyStr,
        extension: str | None = None,
    ) -> Path:
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
            path_to_tmp = Path(tmp.name)
            if isinstance(file_data, bytes):
                path_to_tmp.write_bytes(file_data)
            else:
                path_to_tmp.write_text(str(file_data))
            return path_to_tmp

    def data_path_from_bin_data(
        self,
        raw_file_data: BinaryIO,
        extension: str | None = None,
    ) -> Path:
        return self.data_path_from_data(raw_file_data.read(), extension)

    @classmethod
    @contextmanager
    def temporary_file(
        cls, data_path_fn: Callable[[], Path]
    ) -> Generator[Path, None, None]:
        tmp_path: Path | None = None
        try:
            tmp_path = data_path_fn()
            yield tmp_path
        finally:
            try:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            except Exception as e:
                logger.warning("Failed to delete temporary file %s: %s", tmp_path, e)
