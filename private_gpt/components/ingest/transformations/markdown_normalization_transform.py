import re
from collections.abc import Sequence
from typing import Any

from llama_index.core.schema import BaseNode, MetadataMode, TransformComponent


class MarkdownNormalizer:
    def __init__(self, target_indent: int = 2) -> None:
        self.target_indent = target_indent
        self.PATTERNS = {
            "list": r"^(\s*)([\*\-\+]|\d+\.)\s*(.*)$",
            "table": r"^\s*\|.*\|\s*$",
            "heading": r"^\s*(#{1,6})\s*(.*)$",
            "bold": r"^\*\*(.*)\*\*$",
            "bold_line": r"^\s*\*\*([^*]+)\*\*(.*)$",
            "heading_underline_1": r"^=+$",  # Pattern for level 1 header underline
            "heading_underline_2": r"^-+$",  # Pattern for level 2 header underline
            "code_block": r"^```",  # Pattern for code blocks
        }

    def _normalize_indent(self, indent: str) -> int:
        """Convert tabs to spaces and return the length."""
        # Replace each tab with 4 spaces (standard tab size)
        normalized = indent.replace("\t", "    ")
        return len(normalized)

    def _precalculate_levels(self, lines: list[str]) -> dict[int, int]:
        level_mapping = {}
        sorted_indents: list[int] = []

        for line in lines:
            if re.match(self.PATTERNS["bold_line"], line):
                continue

            list_match = re.match(self.PATTERNS["list"], line)
            if list_match:
                raw_indent = self._normalize_indent(list_match.group(1))
                if raw_indent not in level_mapping:
                    sorted_indents = [i for i in sorted_indents if i < raw_indent]
                    current_level = len(sorted_indents)
                    sorted_indents.append(raw_indent)
                    level_mapping[raw_indent] = current_level

        return level_mapping

    def normalize_markdown(self, content: str) -> str:
        if not content:
            return ""

        lines = content.splitlines()
        normalized_lines = []
        level_mapping = self._precalculate_levels(lines)

        i = 0
        in_code_block = False
        code_block_lines: list[str] = []

        while i < len(lines):
            current_line = lines[i]

            # Handle code blocks
            if re.match(self.PATTERNS["code_block"], current_line.strip()):
                if in_code_block:
                    # End of code block
                    in_code_block = False
                    # Process and add the code block content
                    normalized_lines.extend(code_block_lines)
                    code_block_lines = []
                    normalized_lines.append(current_line)
                else:
                    # Start of code block
                    in_code_block = True
                    normalized_lines.append(current_line)
                i += 1
                continue

            if in_code_block:
                # Store code block content, stripping right whitespace only
                code_block_lines.append(current_line.rstrip())
                i += 1
                continue

            # Check for underline headers
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                current_stripped = current_line.strip()

                # Handle bold text with underlines
                bold_match = re.match(self.PATTERNS["bold"], current_stripped)
                if bold_match and re.match(r"^[=\-]+$", next_line):
                    text = bold_match.group(1)
                    if re.match(self.PATTERNS["heading_underline_1"], next_line):
                        normalized_lines.append(f"# **{text}**")
                    else:
                        normalized_lines.append(f"## **{text}**")
                    i += 2
                    continue

                # Handle regular underline headers
                elif (
                    current_stripped
                    and not re.match(  # Current line not empty
                        self.PATTERNS["heading"], current_stripped
                    )
                    and not re.match(  # Not already a header
                        self.PATTERNS["list"], current_stripped
                    )
                    and not re.match(  # Not a list item
                        self.PATTERNS["table"], current_stripped
                    )
                ):  # Not a table row

                    if (
                        re.match(self.PATTERNS["heading_underline_1"], next_line)
                        and len(next_line) > 0
                    ):
                        normalized_lines.append(f"# {current_stripped}")
                        i += 2  # Skip the underline
                        continue
                    elif (
                        re.match(self.PATTERNS["heading_underline_2"], next_line)
                        and len(next_line) > 0
                    ):
                        normalized_lines.append(f"## {current_stripped}")
                        i += 2  # Skip the underline
                        continue

            # Handle lines that starts with bold text
            if re.match(self.PATTERNS["bold_line"], current_line):
                normalized_lines.append(current_line.rstrip())

            # Handle existing patterns
            elif list_match := re.match(self.PATTERNS["list"], current_line):
                raw_indent, bullet, text = list_match.groups()
                raw_indent_length = self._normalize_indent(raw_indent)
                level = level_mapping.get(raw_indent_length, 0)
                normalized_indent = " " * (level * self.target_indent)
                normalized_lines.append(f"{normalized_indent}{bullet} {text.strip()}")

            elif re.match(self.PATTERNS["table"], current_line):
                normalized_lines.append(current_line)

            elif heading_match := re.match(
                self.PATTERNS["heading"], current_line.strip()
            ):
                hash_marks, text = heading_match.groups()
                normalized_lines.append(f"{hash_marks} {text.strip()}")

            else:
                normalized_lines.append(current_line.rstrip())

            i += 1

        normalized_content = "\n".join(normalized_lines)
        normalized_content = re.sub(r"\n{3,}", "\n\n", normalized_content)

        return normalized_content


class MarkdownNormalizerTransform(TransformComponent):
    @classmethod
    def from_defaults(cls) -> "MarkdownNormalizerTransform":
        return cls()

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        normalizer = MarkdownNormalizer()
        for node in nodes:
            content = node.get_content(MetadataMode.NONE)
            normalized_content = normalizer.normalize_markdown(content)
            node.set_content(normalized_content)

        return nodes
