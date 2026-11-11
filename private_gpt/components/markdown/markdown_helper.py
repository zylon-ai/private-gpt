import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from llama_index.core.utils import iter_batch

# Opening or closing line of a fenced code block (``` or ~~~).
_CODE_FENCE_PATTERN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
# Inline code span: a run of backticks closed by a run of the same length.
_CODE_SPAN_PATTERN = re.compile(r"(?<!`)(`+)(?!`).+?(?<!`)\1(?!`)")


class MarkdownHelper:
    @staticmethod
    def _sanity_data(markdown: str) -> str:
        """Process a single chunk of markdown text."""
        processed = markdown

        # Process bullet characters only if they exist
        if any(char in processed for char in "●·•◦▪▫"):
            processed = re.sub(r"[●·•◦▪▫]\s*", "- ", processed)

        # Table rows keep their inline formatting (bold, italic, links, code)
        # intact so it survives HTML conversion and table cell parsing.
        is_table_row = processed.strip().startswith("|")

        # Quick check if any formatting exists
        if not is_table_row and ("*" in processed or "_" in processed):
            # `_` only delimits emphasis at a word boundary, never inside a word
            # such as snake_case or MAX_TOKENS.

            # 1. Remove bold-italic first (***text*** or ___text___)
            if "***" in processed:
                processed = re.sub(r"\*\*\*(.*?)\*\*\*", r"\1", processed)
            if "___" in processed:
                processed = re.sub(r"(?<!\w)___(.*?)___(?!\w)", r"\1", processed)

            # 2. Remove bold (**text** or __text__)
            if "**" in processed:
                processed = re.sub(r"(?<!\\)\*\*(.*?)(?<!\\)\*\*", r"\1", processed)
            if "__" in processed:
                processed = re.sub(
                    r"(?<![\w\\])__(.*?)(?<!\\)__(?!\w)", r"\1", processed
                )

            # 3. Remove italic (*text* or _text_)
            if "*" in processed:
                processed = re.sub(
                    r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)", r"\1", processed
                )
            if "_" in processed:
                processed = re.sub(r"(?<!\w)_(?!_)(.*?)(?<!_)_(?!\w)", r"\1", processed)

        # Handle list-specific formatting after removing text formatting
        # Remove excessive spaces from headings, lists, and blockquotes
        if "#" in processed:
            processed = re.sub(r"^(#{1,6})\s{2,}", r"\1 ", processed, flags=re.M)
        has_unsorted_list = any(char in processed for char in "-*+")
        has_sorted_list = any(char.isdigit() for char in processed)
        if has_unsorted_list:
            processed = re.sub(r"(\s*[-*+])\s{2,}", r"\1 ", processed)
        if has_sorted_list:
            processed = re.sub(r"(\s*\d+\.)\s{2,}", r"\1 ", processed)

        # Process blockquotes only if they exist
        if ">" in processed:
            processed = re.sub(r"(^>\s?.+\n)\n+(> )", r"\1\2", processed, flags=re.M)
            processed = re.sub(r"(^> .+)(\n\n+)(> )", r"\1\n\3", processed, flags=re.M)

        # Process links/images only if they exist
        if "[" in processed and "]" in processed and "(" in processed:
            if "!" in processed:
                processed = re.sub(
                    r"!\[\s*(.+?)\s*\]\(\s*(.+?)\s*\)", r"![\1](\2)", processed
                )
            processed = re.sub(
                r"\[\s*(.+?)\s*\]\(\s*(.+?)\s*\)", r"[\1](\2)", processed
            )

        # Process list formatting only if lists exist
        if has_unsorted_list or has_sorted_list:
            # Remove extra spaces and duplicate list tokens
            processed = re.sub(
                r"^(\s*[-*+])\s*[-*+]+\s*", r"\1 ", processed, flags=re.M
            )

            # Remove extra spaces from numbered lists
            if has_sorted_list:
                processed = re.sub(r"^(\s*\d+\.)\s{2,}", r"\1 ", processed, flags=re.M)

                # Handle mixed list tokens only if both types exist
                if has_unsorted_list:
                    processed = re.sub(
                        r"^(\s*)[-*+]\s*(\d+\.)\s*", r"\1\2 ", processed, flags=re.M
                    )
                    processed = re.sub(
                        r"^(\s*\d+\.)\s*[-*+]\s*", r"\1 ", processed, flags=re.M
                    )

        # Only normalize paragraph spacing if multiple lines exist
        if "\n" in processed:
            processed = re.sub(r"([^\n])\n([^\n])", r"\1\n\2", processed)

        return processed

    @staticmethod
    def _safe_sanity_data(markdown: str) -> str:
        """Process a chunk in a safe way."""
        protect, restore = MarkdownHelper._protect_patterns()

        # Protect URIs
        protected = protect(markdown)

        # Process markdown
        processed = MarkdownHelper._sanity_data(protected)

        # Restore URIs
        restored = restore(processed)

        return restored

    @staticmethod
    def _protect_patterns() -> tuple[Callable[[str], str], Callable[[str], str]]:
        """Create functions to protect and restore patterns in text."""
        _placeholders: list[str] = []

        def protect(text: str) -> str:
            """Replace patterns with temporary placeholders."""
            # Match both image syntax and URI patterns
            image_pattern = r"!\[\([^)]+\)\]\([^)]+\)"
            uri_pattern = r"(?<!!)\[([^]]+)\]\(([^)]+)\)"

            # Combined pattern for both images and URIs
            combined_pattern = rf"({image_pattern}|{uri_pattern})"

            def replace(match: re.Match[str]) -> str:
                placeholder = f"PATTERN-PLACEHOLDER-{len(_placeholders)}"
                _placeholders.append(match.group(0))
                return placeholder

            # Inline code is literal text: protect it before anything else
            protected = _CODE_SPAN_PATTERN.sub(replace, text)
            protected = re.sub(combined_pattern, replace, protected)
            return protected

        def restore(text: str) -> str:
            """Restore patterns from placeholders."""
            result = text
            # Newest first: a link may wrap an earlier placeholder (inline code),
            # and PATTERN-PLACEHOLDER-1 is a prefix of PATTERN-PLACEHOLDER-10.
            for i, pattern in reversed(list(enumerate(_placeholders))):
                placeholder = f"PATTERN-PLACEHOLDER-{i}"
                result = result.replace(placeholder, pattern)
            return result

        return protect, restore

    @staticmethod
    def _flag_fenced_code_lines(lines: list[str]) -> list[tuple[str, bool]]:
        """Pair each line with whether it is part of a fenced code block."""
        flagged = []
        fence: str | None = None
        for line in lines:
            match = _CODE_FENCE_PATTERN.match(line)
            if fence is None:
                # A backtick fence cannot have backticks in its info string
                if match and not (
                    match.group(1)[0] == "`" and "`" in line[match.end() :]
                ):
                    fence = match.group(1)
                flagged.append((line, fence is not None))
            else:
                flagged.append((line, True))
                if (
                    match
                    and match.group(1)[0] == fence[0]
                    and len(match.group(1)) >= len(fence)
                    and not line[match.end() :].strip()
                ):
                    fence = None
        return flagged

    @staticmethod
    def _sanity_markdown_batches(
        data: tuple[int, list[tuple[str, bool]]],
    ) -> tuple[int, list[str]]:
        index, lines = data

        processed = []
        for line, is_code in lines:
            # Code is kept as written: `*` and `_` there are not emphasis
            processed.append(
                line if is_code else MarkdownHelper._safe_sanity_data(line)
            )

        return index, processed

    @staticmethod
    def sanitize_markdown(
        markdown: str, chunk_size: int | None = 256, max_workers: int | None = None
    ) -> str:
        """Sanitize markdown content to fix common formatting issues."""
        if not markdown:
            return markdown

        chunk_size = chunk_size or len(markdown)
        if chunk_size < 1:
            raise ValueError("Chunk size must be greater than 0.")

        max_workers = max_workers or 1
        max_workers = min(max_workers, len(markdown) // chunk_size)
        max_workers = max(1, max_workers)

        # Split into chunks while trying to preserve line boundaries
        lines = markdown.splitlines(keepends=True)
        # Fences are tracked over the whole text so blocks can span batches
        flagged_lines = MarkdownHelper._flag_fenced_code_lines(lines)
        batches: list[list[tuple[str, bool]]] = list(
            iter_batch(flagged_lines, chunk_size)
        )

        # Create pool of workers
        processed_chunks: list[tuple[int, list[str]]] = []
        if max_workers == 1:
            for index, batch in enumerate(batches):
                _, processed = MarkdownHelper._sanity_markdown_batches((index, batch))
                processed_chunks.append((index, processed))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for index, batch in executor.map(
                    MarkdownHelper._sanity_markdown_batches, enumerate(batches)
                ):
                    processed_chunks.append((index, batch))

        # Sort processed chunks by index
        processed_chunks.sort(key=lambda x: x[0])

        # Join chunks
        return "".join("".join(lines) for _, lines in processed_chunks)
