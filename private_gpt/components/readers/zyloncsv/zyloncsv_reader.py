from collections.abc import Iterator
from csv import Error as CsvError
from csv import Sniffer
from csv import reader as csv_reader
from pathlib import Path
from typing import Any

from llama_index.core.schema import BaseNode, Document

from private_gpt.components.readers.markdown_table_utils import render_markdown_table
from private_gpt.components.readers.text.text_reader import TextReader


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
        encoding = encoding or "utf-8-sig"
        delimiter = _detect_delimiter(file_path, encoding)
        with open(file_path, newline="", encoding=encoding) as file:
            rows = list(csv_reader(file, delimiter=delimiter))

        if not rows:
            yield Document(
                text="", extra_info=extra_info if extra_info is not None else {}
            )
            return

        yield Document(
            text=render_markdown_table(rows),
            extra_info=extra_info if extra_info is not None else {},
        )
