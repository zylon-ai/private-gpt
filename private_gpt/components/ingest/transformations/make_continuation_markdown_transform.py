import re
from collections.abc import Sequence
from typing import Any

from llama_index.core.schema import BaseNode, MetadataMode, TransformComponent


class MarkdownMerger:
    def __init__(self) -> None:
        self.PATTERNS = {
            "list": r"^(\s*[-*+]|\d+\.)\s",
            "table": r"^\|",
            "code_block": r"^```",
            "heading": r"^#{1,6}\s",
            "table_separator": r"^\|[-:|\s]+$",
        }

    def merge_markdown_documents(self, markdown_pages: list[str]) -> list[str]:
        """Merge multiple Markdown documents into a single continuous document.

        :param markdown_pages: List of Markdown strings
        :return: List of merged Markdown strings
        """
        if not markdown_pages:
            return []

        merged_pages = [markdown_pages[0].strip()]
        for next_page in markdown_pages[1:]:
            next_page = next_page.strip()
            last_page = merged_pages[-1]

            merged_content, remaining_content = self._merge_pages(last_page, next_page)
            if remaining_content:
                merged_pages[-1] = merged_content
                merged_pages.append(remaining_content)
            else:
                merged_pages[-1] = merged_content

        return merged_pages

    def _is_compatible_table_structure(self, header1: str, header2: str) -> bool:
        cols1 = [col.strip() for col in header1.strip("|").split("|")]
        cols2 = [col.strip() for col in header2.strip("|").split("|")]

        return len(cols1) == len(cols2)

    def _is_same_table_structure(self, header1: str, header2: str) -> bool:
        cols1 = [col.strip() for col in header1.strip("|").split("|")]
        cols2 = [col.strip() for col in header2.strip("|").split("|")]

        return cols1 == cols2

    def _extract_table_lines(self, lines: list[str]) -> list[tuple[list[str], int]]:
        tables = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if re.match(self.PATTERNS["table"], line):
                table_lines = []
                start_index = index
                header_line = lines[index]
                table_lines.append(header_line)
                index += 1

                # Check for table separator
                if index < len(lines) and re.match(
                    self.PATTERNS["table_separator"], lines[index]
                ):
                    table_lines.append(lines[index])
                    index += 1
                else:
                    # Not a valid table
                    index = start_index + 1
                    continue

                # Collect table rows
                while index < len(lines) and re.match(
                    self.PATTERNS["table"], lines[index]
                ):
                    table_lines.append(lines[index])
                    index += 1

                tables.append((table_lines, index))
            else:
                index += 1
        return tables

    def _merge_pages(self, current_content: str, next_page: str) -> tuple[str, str]:
        """Merge two pages of Markdown, handling different element continuations.

        :param current_content: Current Markdown content
        :param next_page: Next page's Markdown content
        :return: Tuple of (merged content, remaining content)
        """
        current_lines = current_content.splitlines()
        next_lines = next_page.splitlines()

        # Check if the current lines have an odd number of code block markers
        block_elements = sum("```" in line for line in current_lines)
        inside_code_block = block_elements % 2 == 1

        # Extract the last table from current content
        current_tables = self._extract_table_lines(current_lines)
        last_table = current_tables[-1] if current_tables else None
        last_table_header = last_table[0][0] if last_table else None

        while next_lines:
            first_next_line = next_lines[0]

            if inside_code_block:
                if first_next_line.startswith("```"):
                    current_lines.append(next_lines.pop(0))
                    inside_code_block = False
                else:
                    current_lines.append(next_lines.pop(0))
            elif last_table and re.match(self.PATTERNS["table"], first_next_line):
                extracted_tables = self._extract_table_lines(next_lines)
                if extracted_tables:
                    # Handle a new table on the next page
                    table_lines, table_end_index = extracted_tables[0]
                    table_header = table_lines[0]

                    if last_table_header and not self._is_same_table_structure(
                        last_table_header, table_header
                    ):
                        break
                    else:
                        current_lines.extend(table_lines[2:])
                        next_lines = next_lines[table_end_index:]
                        continue
                elif last_table_header:
                    # Handle continuation of the last table
                    continuation_rows = [
                        line
                        for line in next_lines
                        if re.match(self.PATTERNS["table"], line)
                    ]
                    if continuation_rows and all(
                        self._is_compatible_table_structure(last_table_header, line)
                        for line in continuation_rows
                    ):
                        current_lines.extend(continuation_rows)
                        next_lines = next_lines[len(continuation_rows) :]
                        continue
                else:
                    break
            elif last_table and not extracted_tables:
                # Handle continuation of the last table
                continuation_rows = [
                    line
                    for line in next_lines
                    if re.match(self.PATTERNS["table"], line)
                ]
                if continuation_rows:
                    current_lines.extend(continuation_rows)
                    next_lines = next_lines[len(continuation_rows) :]
                    continue
                else:
                    break
            elif re.match(self.PATTERNS["table"], first_next_line):
                # Handle new tables on the next page
                extracted_tables = self._extract_table_lines(next_lines)
                if extracted_tables:
                    table_lines, table_end_index = extracted_tables[0]
                    current_lines.append("\n".join(table_lines))
                    next_lines = next_lines[table_end_index:]
                    continue
                else:
                    break
            elif self._is_list_continuation(current_lines, first_next_line):
                current_lines.append(next_lines.pop(0))
            elif self._is_paragraph_continuation(current_lines, first_next_line):
                current_lines[-1] += " " + next_lines.pop(0)
            else:
                break

        return "\n".join(current_lines), "\n".join(next_lines)

    def _is_list_continuation(self, current_lines: list[str], next_line: str) -> bool:
        return bool(
            current_lines
            and re.match(self.PATTERNS["list"], current_lines[-1])
            and re.match(self.PATTERNS["list"], next_line)
        )

    def _is_paragraph_continuation(
        self, current_lines: list[str], next_line: str
    ) -> bool:
        last_line = current_lines[-1] if current_lines else ""
        return bool(
            last_line
            and next_line
            and not any(
                re.match(pattern, next_line) for pattern in self.PATTERNS.values()
            )
            and not last_line.endswith(("```", "|", "#"))
        )


class MakeContinuationMarkdownTransform(TransformComponent):
    """Merge multiple Markdown documents into a single continuous document."""

    @classmethod
    def from_defaults(cls) -> "MakeContinuationMarkdownTransform":
        return cls()

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        merger = MarkdownMerger()
        texts = [node.get_content(MetadataMode.NONE) for node in nodes]
        merged_markdown = merger.merge_markdown_documents(texts)

        for node, merged_text in zip(nodes, merged_markdown, strict=False):
            node.set_content(merged_text)

        return nodes
