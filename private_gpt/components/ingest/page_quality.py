"""Cheap, regex-based quality checks for pdf-inspector's per-page markdown.

Each check inspects a single page's markdown and decides whether that page
looks mis-extracted for one quality dimension (broken table, fragmented
figure text, etc.). ``QualityFilter`` aggregates a check across all pages of
a document into a single bad-page ratio, and ``evaluate_document_quality``
runs a list of filters and returns the first one that fails its threshold.
This lets callers (e.g. ``HybridPdfReader``) treat "is this document good
enough for the hybrid pipeline" as one call, and add new heuristics as new
``QualityFilter`` entries instead of hand-rolling a new ratio/threshold
check for each one.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

_SEPARATOR_ROW_RE = re.compile(r"\|[\s:|-]+\|")
_WORD_RE = re.compile(r"\S+")
_ALNUM_RE = re.compile(r"[^\W_]")
_DECIMAL_POINT_RE = re.compile(r"\d\.\d")
_LONG_CELL_WORD_COUNT = 5
_GLUED_DECIMAL_POINTS_COUNT = 2


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
    markdown: str,
    avg_words_per_cell_threshold: float,
    pct_long_cells_threshold: float,
) -> bool:
    """Whether a page's markdown table looks like mis-segmented prose.

    A markdown table is considered "broken" when pdf-inspector has chopped
    narrative/paragraph text into table cells (each cell a sentence
    fragment) rather than extracted genuine tabular data (short
    label/value cells).
    """
    cells = parse_markdown_table_cells(markdown)
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


def has_table(markdown: str) -> bool:
    return bool(parse_markdown_table_cells(markdown))


def empty_cell_ratio(markdown: str) -> float:
    """Fraction of a page's markdown table cells that are empty.

    Charts (bar/line plots) mis-extracted as a markdown table tend to
    produce one row per bar/label with the value in a single column and all
    other columns empty, unlike real data tables which are mostly filled in.
    """
    total = 0
    empty = 0
    for line in markdown.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if _SEPARATOR_ROW_RE.fullmatch(stripped):
            continue
        row_cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        total += len(row_cells)
        empty += sum(1 for cell in row_cells if not cell)
    return (empty / total) if total else 0.0


def is_sparse_chart_table(markdown: str, empty_cell_ratio_threshold: float) -> bool:
    """Whether a page's markdown table looks like a chart mis-extracted as a table."""
    return empty_cell_ratio(markdown) >= empty_cell_ratio_threshold


def glued_numbers_ratio(markdown: str) -> float:
    """Fraction of table cells containing 2+ separate decimal numbers glued together.

    When a chart/table's column separators are lost, adjacent numeric values
    end up concatenated into a single cell (e.g. "0.3830.207" instead of two
    cells "0.383" and "0.207"), which a well-formed data table cell never
    exhibits.
    """
    total = 0
    glued = 0
    for line in markdown.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if _SEPARATOR_ROW_RE.fullmatch(stripped):
            continue
        row_cells = [
            cell.strip() for cell in stripped.strip("|").split("|") if cell.strip()
        ]
        for cell in row_cells:
            total += 1
            if len(_DECIMAL_POINT_RE.findall(cell)) >= _GLUED_DECIMAL_POINTS_COUNT:
                glued += 1
    return (glued / total) if total else 0.0


def has_glued_numbers(markdown: str, glued_numbers_ratio_threshold: float) -> bool:
    """Whether a page's markdown table has cells with concatenated numbers."""
    return glued_numbers_ratio(markdown) >= glued_numbers_ratio_threshold


def fragmented_text_ratio(markdown: str, max_token_chars: int = 2) -> float:
    """Fraction of tokens that are alnum-fragments of at most `max_token_chars`.

    Figures (scatter plots, charts) extracted as text tend to have numbers
    and words chopped up character-by-character (e.g. "2 8. 5 9" instead of
    "28.59", "Qw en3-" instead of "Qwen3-"), which produces a markdown blob
    with an abnormally high ratio of very short tokens compared to normal
    prose or even normal tables.
    """
    tokens = _WORD_RE.findall(markdown)
    if not tokens:
        return 0.0
    short = sum(
        1
        for token in tokens
        if _ALNUM_RE.search(token) and len(_ALNUM_RE.findall(token)) <= max_token_chars
    )
    return short / len(tokens)


def is_fragmented_text(markdown: str, short_token_ratio_threshold: float) -> bool:
    """Whether a page's markdown looks like figure text mangled into fragments."""
    return fragmented_text_ratio(markdown) >= short_token_ratio_threshold


def has_any_text(markdown: str) -> bool:
    return bool(markdown.strip())


@dataclass(frozen=True)
class QualityIssue:
    """A quality dimension that failed for a document."""

    name: str
    reason: str


PageChecker = Callable[[str], bool]


@dataclass(frozen=True)
class QualityFilter:
    """A single document-level quality dimension.

    `applies` decides whether a page counts towards this filter's ratio at
    all (e.g. only pages containing a parseable table, for the broken-table
    filter). `is_bad` decides whether a page that applies is "bad" for this
    dimension. The document fails this filter when the ratio of bad pages
    among applicable pages exceeds `threshold`.
    """

    name: str
    applies: PageChecker
    is_bad: PageChecker
    threshold: float

    def evaluate(self, pages_markdown: list[str]) -> QualityIssue | None:
        applicable = [md for md in pages_markdown if self.applies(md)]
        if not applicable:
            return None

        bad_count = sum(1 for md in applicable if self.is_bad(md))
        ratio = bad_count / len(applicable)
        if ratio <= self.threshold:
            return None

        return QualityIssue(
            name=self.name,
            reason=(
                f"{ratio:.2%} of applicable pages failed the '{self.name}' "
                f"quality check, above threshold ({self.threshold:.2%})."
            ),
        )


def evaluate_document_quality(
    pages_markdown: list[str],
    filters: list[QualityFilter],
) -> QualityIssue | None:
    """Run every filter in order and return the first quality issue found."""
    for quality_filter in filters:
        issue = quality_filter.evaluate(pages_markdown)
        if issue is not None:
            return issue
    return None
