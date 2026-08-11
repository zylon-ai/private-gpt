import re

_SEPARATOR_ROW_RE = re.compile(r"\|[\s:|-]+\|")
_WORD_RE = re.compile(r"\S+")
_LONG_CELL_WORD_COUNT = 5


def parse_markdown_table_cells(markdown: str) -> list[str]:
    """Extract all data cells (excluding separator rows) from markdown tables."""
    cells = []
    for line in markdown.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if _SEPARATOR_ROW_RE.fullmatch(stripped):
            continue
        row_cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        cells.extend(cell for cell in row_cells if cell)
    return cells


def is_broken_table(
    cells: list[str],
    avg_words_per_cell_threshold: float,
    pct_long_cells_threshold: float,
) -> bool:
    """Whether markdown table cells look like mis-segmented prose, not real data.

    A markdown table is considered "broken" when pdf-inspector has chopped
    narrative/paragraph text into table cells (each cell a sentence
    fragment) rather than extracted genuine tabular data (short
    label/value cells).
    """
    if not cells:
        return False

    word_counts = [len(_WORD_RE.findall(cell)) for cell in cells]
    avg_words_per_cell = sum(word_counts) / len(word_counts)
    pct_long_cells = sum(
        1 for count in word_counts if count >= _LONG_CELL_WORD_COUNT
    ) / len(cells)

    return (
        avg_words_per_cell >= avg_words_per_cell_threshold
        or pct_long_cells >= pct_long_cells_threshold
    )
