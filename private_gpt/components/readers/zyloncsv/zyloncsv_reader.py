from collections.abc import Iterator
from csv import Error as CsvError
from csv import Sniffer
from csv import reader as csv_reader
from pathlib import Path
from typing import Any

from llama_index.core.schema import BaseNode, Document

from private_gpt.components.readers.markdown_table_utils import render_markdown_table
from private_gpt.components.readers.text.text_reader import TextReader
from private_gpt.settings.settings import settings


def _detect_delimiter(file_path: Path, encoding: str, sample_size: int = 4096) -> str:
    """Sniff the delimiter from a small sample. Falls back to comma.

    Most ".csv" files really are comma-separated, but plenty (mostly from
    European locales) use ";" instead - a cheap one-off sniff on a few KB
    avoids silently mis-parsing those into a single-column table.
    """
    try:
        with open(file_path, encoding=encoding) as file:
            sample = file.read(sample_size)
        return Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except (CsvError, OSError):
        return ","


_ROWS_PER_CHUNK = 5000
"""Rows buffered in memory before a chunk is turned into a Document.

Reading and rendering the whole CSV in one shot means holding the raw rows,
the formatted rows and the rendered Markdown string all at once - for a
large file that multiplies peak memory several times over. Streaming the
file row by row and flushing a Document every `_ROWS_PER_CHUNK` rows keeps
peak memory bounded regardless of file size.
"""


class ZylonCsvReader(TextReader):
    """Fast CSV -> Markdown reader.

    Converts a CSV straight into a Markdown table using the stdlib `csv`
    module (a C extension): no pandas dtype inference, no per-cell Python
    object churn. For large CSVs this is several times faster than a
    pandas-based conversion and it never mangles values (e.g. an id like
    "007" stays "007" instead of becoming "7.0").
    """

    def lazy_document_load(
        self,
        file_path: Path,
        encoding: str | None = None,
        extra_info: dict[str, Any] | None = None,
    ) -> Iterator[BaseNode]:
        file_size = file_path.stat().st_size
        max_size = settings().data.limits.max_file_size
        if file_size > max_size:
            raise ValueError(
                f"File {file_path} ({file_size} bytes) exceeds the maximum "
                f"allowed size of {max_size} bytes for CSV ingestion"
            )

        encoding = encoding or "utf-8-sig"
        delimiter = _detect_delimiter(file_path, encoding)
        extra_info = extra_info if extra_info is not None else {}

        with open(file_path, newline="", encoding=encoding) as file:
            rows = csv_reader(file, delimiter=delimiter)
            try:
                header = next(rows)
            except StopIteration:
                yield Document(text="", extra_info=extra_info)
                return

            chunk: list[list[str]] = []
            chunk_emitted = False
            for row in rows:
                chunk.append(row)
                if len(chunk) >= _ROWS_PER_CHUNK:
                    yield Document(
                        text=render_markdown_table([header, *chunk]),
                        extra_info=extra_info,
                    )
                    chunk_emitted = True
                    chunk = []

            if chunk or not chunk_emitted:
                yield Document(
                    text=render_markdown_table([header, *chunk]),
                    extra_info=extra_info,
                )
