import re
from collections.abc import Sequence
from typing import Any

import numpy as np
from llama_index.core.schema import BaseNode, MetadataMode, TransformComponent


class RemoveHeaderAndFooterTransform(TransformComponent):
    """Remove headers and footers from PDF content using fuzzy matching."""

    candidate_window: int = 10
    remove_header: bool = True
    remove_footer: bool = True
    keep_initial_header: bool = True
    keep_latest_footer: bool = True

    @classmethod
    def from_defaults(
        cls,
    ) -> "RemoveHeaderAndFooterTransform":
        return cls()

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        if not nodes:
            return nodes

        texts = [node.get_content(MetadataMode.NONE) for node in nodes]
        pages = [text.splitlines() for text in texts]
        cleaned_pages = self.remove_header_footer(pages)

        for node, cleaned_page in zip(nodes, cleaned_pages, strict=False):
            node.set_content("\n".join(cleaned_page))
        return nodes

    def _compare(self, a: str, b: str) -> float:
        """Compare two strings with fuzzy matching."""
        if a == b:
            return 1.0
        a = re.sub(r"\d", "@", a)
        b = re.sub(r"\d", "@", b)
        count = sum(1 for x, y in zip(a, b, strict=False) if x == y)
        max_length = max(len(a), len(b))
        return count / max_length if max_length > 0 else 0

    def compute_logarithmic_weights(
        self, num_candidates: int, start: float = 1.0, end: float = 0.5
    ) -> list[float]:
        """Compute logarithmic weights for header/footer detection."""
        if num_candidates == 0:
            return []
        log_weights = np.logspace(0, 1, num_candidates, base=10, endpoint=True)
        log_weights = log_weights / max(log_weights)
        scaled_weights = start - log_weights * (start - end)
        return list(scaled_weights)

    def _remove_header(
        self, pages: list[list[str]], header_candidates: list[list[str]], win: int
    ) -> list[list[str]]:
        """Remove headers from pages."""
        header_weights = self.compute_logarithmic_weights(len(header_candidates[0]))

        for i, candidate in enumerate(header_candidates):
            temp_candidates = header_candidates[
                max(i - win, 0) : min(i + win, len(header_candidates))
            ]
            maxlen = len(max(temp_candidates, key=len)) if temp_candidates else 0

            # Pad shorter lists
            padded_candidates = [
                sublist + [""] * (maxlen - len(sublist)) for sublist in temp_candidates
            ]

            detected = []
            for j, cn in enumerate(candidate):
                score = 0.0
                try:
                    # Safely transpose and compare candidates
                    cmp = list(zip(*padded_candidates, strict=False))[j]
                    score = sum(
                        self._compare(cn, cm) * header_weights[j] for cm in cmp
                    ) / len(cmp)
                except (IndexError, ZeroDivisionError):
                    score = header_weights[j] if j < len(header_weights) else 0

                if score > 0.5:
                    detected.append(cn)
                else:
                    break

            # Remove detected headers from pages
            if not self.keep_initial_header or i != 0:
                for d in detected:
                    while d in pages[i][: len(header_candidates[0])]:
                        pages[i].remove(d)

        return pages

    def _remove_footer(
        self, pages: list[list[str]], footer_candidates: list[list[str]], win: int
    ) -> list[list[str]]:
        footer_weights = list(
            reversed(self.compute_logarithmic_weights(len(footer_candidates[0])))
        )

        for i, candidate in enumerate(footer_candidates):
            temp_candidates = footer_candidates[
                max(i - win, 0) : min(i + win + 1, len(footer_candidates))
            ]
            maxlen = max((len(sublist) for sublist in temp_candidates), default=0)

            # Pad shorter lists
            padded_candidates = [
                [""] * (maxlen - len(sublist)) + sublist for sublist in temp_candidates
            ]
            padded_candidates_reversed = list(reversed(padded_candidates))

            detected = []
            for j in range(len(candidate) - 1, -1, -1):
                score = 0.0
                cn = candidate[j]
                try:
                    # Safely transpose and compare candidates
                    cmp = list(zip(*padded_candidates_reversed, strict=False))[j]
                    score = sum(
                        self._compare(cn, cm) * footer_weights[j] for cm in cmp
                    ) / len(cmp)
                except (IndexError, ZeroDivisionError):
                    score = footer_weights[j] if 0 <= j < len(footer_weights) else 0

                if score > 0.5:
                    detected.append(cn)
                else:
                    break

            # Remove detected footers from the current page
            if not self.keep_latest_footer or i != len(pages) - 1:
                for footer in detected:
                    pages[i] = [line for line in pages[i] if line != footer]

        return pages

    def remove_header_footer(
        self, pages: list[list[str]], win: int = 2
    ) -> list[list[str]]:
        # Remove header
        if self.remove_header:
            header_candidates = [
                page[: min(self.candidate_window, len(page))] for page in pages if page
            ]
            if header_candidates and len(pages) > 1:
                pages = self._remove_header(pages, header_candidates, win)

        # Remove footer
        if self.remove_footer:
            footer_candidates = [
                page[-min(self.candidate_window, len(page)) :] for page in pages if page
            ]
            if footer_candidates and len(pages) > 1:
                pages = self._remove_footer(pages, footer_candidates, win)

        return pages
