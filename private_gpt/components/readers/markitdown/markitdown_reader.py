import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from llama_index.core.schema import BaseNode, Document

from private_gpt.components.readers.text.text_reader import TextReader

logger = logging.getLogger(__name__)


class MarkItDownReader(TextReader):
    def lazy_document_load(
        self,
        file_path: Path,
        encoding: str | None = None,
        extra_info: dict[str, Any] | None = None,
    ) -> Iterator[BaseNode]:
        from markitdown import (  # type: ignore[import-not-found]  # ty:ignore[unresolved-import]
            MarkItDown,
            StreamInfo,
        )

        logger.debug("Starting MarkItDown conversion of file: %s", file_path)
        converter = MarkItDown(
            enable_builtins=True,
            enable_plain=True,
        )
        result = converter.convert(
            source=file_path,
            stream_info=StreamInfo(
                charset=encoding,
            ),
        )
        content = result.markdown or result.text_content or ""
        logger.debug("Finished MarkItDown conversion of file: %s", file_path)

        yield Document(
            text=content,
            extra_info=extra_info if extra_info is not None else {},
        )
