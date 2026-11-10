import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from llama_index.core.schema import BaseNode, Document

from private_gpt.components.readers.text.text_reader import TextReader

# The characters a field separator can be. Unrestricted, csv.Sniffer picks a
# letter in a one-column file and splits the header "Note" into "No" and "e".
DELIMITERS = ",;\t|"


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
            dialect = sniffer.sniff(sample, delimiters=DELIMITERS)
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

            # Read every cell as the text it holds: inferred dtypes would turn
            # "01067" into "1067.0", and pandas' default NA values would turn
            # empty cells, "NA" and "N/A" into "nan".
            for chunk in pd.read_csv(
                file_path,
                # No detected separator means one column. sep=None would make
                # pandas sniff again, without the restriction above.
                sep=delimiter or ",",
                encoding=encoding or "utf-8",
                on_bad_lines="warn",
                chunksize=chunksize,
                dtype=str,
                keep_default_na=False,
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
