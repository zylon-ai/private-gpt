from collections.abc import Iterator
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from llama_index.core.schema import BaseNode, Document
from python_calamine import CalamineError, CalamineSheet, CalamineWorkbook

from private_gpt.components.readers.markdown_table_utils import render_markdown_table
from private_gpt.components.readers.text.text_reader import TextReader
from private_gpt.settings.settings import settings

_TEMPORAL_TYPES = (datetime, date, time)

_ROWS_PER_CHUNK = 5000
"""Rows buffered per table block before it's rendered and released.

Without this, a sheet with no blank rows (a single big table) keeps
accumulating `current_rows` for the entire sheet before rendering it to
Markdown - holding the raw cell values and the rendered string for the
whole sheet in memory at once. Flushing every `_ROWS_PER_CHUNK` rows bounds
that regardless of sheet size.
"""


def _format_cell_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, _TEMPORAL_TYPES):
        return value.isoformat()
    if value is None:
        return ""
    return str(value)


def _row_merges_by_row(
    merges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> dict[int, list[tuple[int, int, int, int]]]:
    # merged_cell_ranges is 0-based (row, col) with an inclusive end coordinate.
    row_merges: dict[int, list[tuple[int, int, int, int]]] = {}
    for (row_start, col_start), (row_end, col_end) in merges:
        entry = (col_start, col_end, row_start, col_start)
        for row in range(row_start, row_end + 1):
            row_merges.setdefault(row, []).append(entry)
    return row_merges


def _is_blank_row(values: list[Any]) -> bool:
    return all(v is None or v == "" for v in values)


class _TableAccumulator:
    """Buffers formatted rows for one table block and renders it in chunks.

    A block gets rendered to Markdown and released every `_ROWS_PER_CHUNK`
    rows (repeating the block's header on each continuation chunk), instead
    of only when a blank row or the end of the sheet is reached. Without
    this, a sheet that is one giant table with no blank rows would hold
    every row - raw and formatted - in memory until the whole sheet had
    been read.
    """

    def __init__(self, tables: list[str]) -> None:
        self._tables = tables
        self._rows: list[list[str]] = []
        self._header: list[str] | None = None

    def append(self, row: list[str]) -> None:
        if not self._rows:
            self._header = row
        self._rows.append(row)
        if len(self._rows) > _ROWS_PER_CHUNK:
            self._tables.append(render_markdown_table(self._rows))
            self._rows = [self._header]  # type: ignore[list-item]

    def flush(self) -> None:
        if self._rows:
            self._tables.append(render_markdown_table(self._rows))
        self._rows = []
        self._header = None


def _sheet_to_markdown(sheet: CalamineSheet) -> str | None:
    rows = sheet.to_python(skip_empty_area=False)
    if not rows:
        return None

    merges = sheet.merged_cell_ranges or []
    format_value = _format_cell_value

    tables: list[str] = []
    accumulator = _TableAccumulator(tables)

    if not merges:
        for row in rows:
            if _is_blank_row(row):
                accumulator.flush()
                continue
            accumulator.append([format_value(v) for v in row])
        accumulator.flush()
        return "\n\n".join(tables) if tables else None

    row_merges = _row_merges_by_row(merges)
    max_merge_col_end = max(
        col_end for (_row_start, _col_start), (_row_end, col_end) in merges
    )
    sheet_width = max(len(rows[0]), max_merge_col_end + 1)
    merge_values: dict[tuple[int, int], Any] = {}

    for row_index, row in enumerate(rows):
        merges_here = row_merges.get(row_index)
        if merges_here is None and len(row) >= sheet_width:
            if _is_blank_row(row):
                accumulator.flush()
                continue
            accumulator.append([format_value(v) for v in row])
            continue

        values: list[Any] = list(row)
        if len(values) < sheet_width:
            values.extend([""] * (sheet_width - len(values)))

        for col_start, col_end, top_row, top_col in merges_here or ():
            if row_index == top_row:
                merge_values[(top_row, top_col)] = values[top_col]
            fill_value = merge_values.get((top_row, top_col))
            for col in range(col_start, col_end + 1):
                if row_index == top_row and col == top_col:
                    continue
                values[col] = fill_value

        if _is_blank_row(values):
            accumulator.flush()
            continue
        accumulator.append([format_value(v) for v in values])

    accumulator.flush()

    return "\n\n".join(tables) if tables else None


class ZylonXlsxReader(TextReader):
    """Fast xlsx -> Markdown reader."""

    def lazy_document_load(
        self,
        file_path: Path,
        encoding: str | None = None,
        extra_info: dict[str, Any] | None = None,
    ) -> Iterator[BaseNode]:
        del encoding
        file_size = file_path.stat().st_size
        max_size = settings().data.limits.max_file_size
        if file_size > max_size:
            raise ValueError(
                f"File {file_path} ({file_size} bytes) exceeds the maximum "
                f"allowed size of {max_size} bytes for XLSX ingestion"
            )

        try:
            workbook = CalamineWorkbook.from_path(str(file_path))
        except (OSError, CalamineError) as exc:
            raise ValueError(f"Could not open xlsx file {file_path}: {exc}") from exc

        try:
            sections: list[str] = []
            for sheet_name in workbook.sheet_names:
                sheet = workbook.get_sheet_by_name(sheet_name)
                table = _sheet_to_markdown(sheet)
                if table is None:
                    continue
                sections.append(f"## {sheet_name}\n\n{table}")
        finally:
            workbook.close()

        yield Document(
            text="\n\n".join(sections),
            extra_info=extra_info if extra_info is not None else {},
        )
