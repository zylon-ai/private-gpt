import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from llama_index.core.schema import BaseNode, Document

from private_gpt.components.readers.text.text_reader import TextReader


class DelimiterTextReader(TextReader):
    """Reader for handling delimited files (CSV, TSV, etc.)."""

    def __init__(self) -> None:
        super().__init__()

    def _detect_delimiter_and_header(
        self, file_path: Path, sample_size: int = 4096
    ) -> tuple[str | None, bool | None]:
        """Detect the delimiter of a file by analyzing a sample.

        Args:
            file_path: Path to the delimited file
            sample_size: Number of bytes to sample for detection

        Returns:
            str: Detected delimiter
        """
        with open(file_path) as file:
            sample = file.read(sample_size)

        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
            has_header = sniffer.has_header(sample)
            return str(dialect.delimiter), has_header
        except csv.Error:
            return None, None

    def lazy_document_load(
        self,
        file_path: Path,
        encoding: str | None = None,
        extra_info: dict[str, Any] | None = None,
    ) -> Iterator[BaseNode]:
        delimiter, _ = self._detect_delimiter_and_header(file_path)

        chunksize = 100
        markdown_lines = []
        first_chunk = True
        try:

            def format_row(row: list[str]) -> str:
                result = "| " + " | ".join(str(cell).strip() for cell in row) + " |"
                return " ".join(result.split())

            for chunk in pd.read_csv(
                file_path,
                sep=delimiter,
                encoding=encoding or "utf-8",
                on_bad_lines="warn",
                chunksize=chunksize,
            ):
                if first_chunk:
                    # Generate header and separator rows
                    header_row = format_row(list(chunk.columns))
                    separator_row = "| " + " | ".join("-" for _ in chunk.columns) + " |"
                    markdown_lines.append(header_row)
                    markdown_lines.append(separator_row)
                    first_chunk = False

                # Convert chunk to strings and format rows
                chunk_str = chunk.astype(str)
                for row in chunk_str.values.tolist():
                    markdown_lines.append(format_row(row))

            # Join all Markdown lines into a single string.
            markdown_content = "\n".join(markdown_lines)
            yield Document(
                text=markdown_content,
                extra_info=extra_info if extra_info is not None else {},
            )

        except Exception as e:
            raise ValueError(f"Error reading delimited file: {e}") from e
