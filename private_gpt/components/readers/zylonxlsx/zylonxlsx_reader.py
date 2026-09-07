from collections.abc import Iterator
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from llama_index.core.schema import BaseNode, Document
from python_calamine import CalamineError, CalamineSheet, CalamineWorkbook

from private_gpt.components.readers.markdown_table_utils import render_markdown_table
from private_gpt.components.readers.text.text_reader import TextReader

_TEMPORAL_TYPES = (datetime, date, time)


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


def _sheet_to_markdown(sheet: CalamineSheet) -> str | None:
    rows = sheet.to_python(skip_empty_area=False)
    if not rows:
        return None

    merges = sheet.merged_cell_ranges or []
    format_value = _format_cell_value

    tables: list[str] = []
    current_rows: list[list[str]] = []

    if not merges:
        for row in rows:
            if _is_blank_row(row):
                if current_rows:
                    tables.append(render_markdown_table(current_rows))
                    current_rows = []
                continue
            current_rows.append([format_value(v) for v in row])
        if current_rows:
            tables.append(render_markdown_table(current_rows))
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
                if current_rows:
                    tables.append(render_markdown_table(current_rows))
                    current_rows = []
                continue
            current_rows.append([format_value(v) for v in row])
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
            if current_rows:
                tables.append(render_markdown_table(current_rows))
                current_rows = []
            continue
        current_rows.append([format_value(v) for v in values])

    if current_rows:
        tables.append(render_markdown_table(current_rows))

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
