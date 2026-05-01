"""Semantic chunking and hash computation module.

This module implements paragraph-based (semantic) chunking to avoid the
"avalanche effect" described in the thesis: with fixed-size chunking, a small
text insertion shifts all subsequent chunks, causing unnecessary re-embeddings.

By splitting on paragraph boundaries (double newlines, headings, etc.), each
chunk is a self-contained semantic unit. A SHA-256 hash is computed per chunk
so that changes can be detected efficiently.

References (from thesis):
- Semantic chunking via breakpoints (Qu et al., 2024)
- Hash-based change detection (LangChain Sync Vector Stores, 2023)
- Avalanche effect in fixed-size chunking
  (thesis: State of the art - Chunking strategies)
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HashedChunk:
    """A chunk of text with its computed hash and metadata.

    Attributes:
        chunk_index: Position of the chunk within the document (0-based).
        text: The raw text content of the chunk.
        content_hash: SHA-256 hex digest of the normalised text.
        metadata: Arbitrary metadata carried along (e.g. file_name, page).
    """

    chunk_index: int
    text: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HashedChunk):
            return NotImplemented
        return self.content_hash == other.content_hash

    def __hash__(self) -> int:
        return hash(self.content_hash)


class SemanticChunker:
    """Splits text into semantic chunks based on paragraph boundaries.

    Unlike fixed-size chunking (e.g. 1024 tokens with 20-token overlap),
    semantic chunking splits on natural boundaries:
    - Double newlines (paragraph breaks)
    - Markdown/reStructuredText headings
    - Section separators (---, ===, etc.)

    This prevents the avalanche/domino effect where a small edit shifts
    all subsequent fixed-size chunks, forcing unnecessary re-embeddings.

    Parameters:
        min_chunk_size: Minimum characters for a chunk (smaller chunks are
                        merged with the next one).
        max_chunk_size: Maximum characters for a chunk (larger chunks are
                        split at sentence boundaries).
    """

    # Regex patterns for paragraph boundaries
    PARAGRAPH_SPLIT_PATTERN = re.compile(
        r"(?:"
        r"\n\s*\n"  # Double newline (paragraph break)
        r"|(?=^#{1,6}\s)"  # Markdown heading
        r"|(?=^={3,}$)"  # === separator
        r"|(?=^-{3,}$)"  # --- separator
        r")",
        re.MULTILINE,
    )

    # Sentence boundary for splitting oversized chunks
    SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 3000,
    ) -> None:
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk_text(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[HashedChunk]:
        """Split text into semantic chunks and compute hashes.

        Args:
            text: The full document text to split.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            A list of HashedChunk objects with computed hashes.
        """
        if metadata is None:
            metadata = {}

        # Step 1: Split on paragraph boundaries
        raw_chunks = self.PARAGRAPH_SPLIT_PATTERN.split(text)

        # Step 2: Clean and filter empty chunks
        raw_chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]

        # Step 3: Merge small chunks and split oversized ones
        merged_chunks = self._merge_and_split(raw_chunks)

        # Step 4: Compute hashes
        hashed_chunks = []
        for idx, chunk_text in enumerate(merged_chunks):
            content_hash = self._compute_hash(chunk_text)
            hashed_chunks.append(
                HashedChunk(
                    chunk_index=idx,
                    text=chunk_text,
                    content_hash=content_hash,
                    metadata={**metadata, "chunk_index": idx},
                )
            )

        logger.debug(
            "Chunked text into %d semantic chunks (from %d characters)",
            len(hashed_chunks),
            len(text),
        )
        return hashed_chunks

    def _merge_and_split(self, chunks: list[str]) -> list[str]:
        """Merge chunks that are too small and split those that are too large.

        Small chunks (< min_chunk_size) are merged with the following chunk
        to avoid creating trivially small embeddings. Large chunks
        (> max_chunk_size) are split at sentence boundaries to stay within
        the embedding model's context window.
        """
        result: list[str] = []
        buffer = ""

        for chunk in chunks:
            candidate = buffer + "\n\n" + chunk if buffer else chunk

            if len(candidate) < self.min_chunk_size:
                # Too small - keep buffering
                buffer = candidate
            elif len(candidate) > self.max_chunk_size:
                # If we have a buffer that's large enough on its own, flush it
                if buffer and len(buffer) >= self.min_chunk_size:
                    result.extend(self._split_oversized(buffer))
                    buffer = ""
                    # Now handle the current chunk
                    if len(chunk) > self.max_chunk_size:
                        result.extend(self._split_oversized(chunk))
                    else:
                        buffer = chunk
                else:
                    # The combined text is too large; split it
                    result.extend(self._split_oversized(candidate))
                    buffer = ""
            else:
                result.append(candidate)
                buffer = ""

        # Don't forget the remaining buffer
        if buffer:
            if result and len(buffer) < self.min_chunk_size:
                # Merge with last chunk if buffer is tiny
                result[-1] = result[-1] + "\n\n" + buffer
            else:
                result.append(buffer)

        return result

    def _split_oversized(self, text: str) -> list[str]:
        """Split a chunk that exceeds max_chunk_size at sentence boundaries."""
        sentences = self.SENTENCE_SPLIT_PATTERN.split(text)
        result: list[str] = []
        current = ""

        for sentence in sentences:
            candidate = (current + " " + sentence).strip() if current else sentence
            if len(candidate) > self.max_chunk_size and current:
                result.append(current.strip())
                current = sentence
            else:
                current = candidate

        if current.strip():
            result.append(current.strip())

        return result

    @staticmethod
    def _compute_hash(text: str) -> str:
        """Compute SHA-256 hash of normalised text.

        Normalisation:
        - Strip leading/trailing whitespace
        - Collapse multiple whitespace to single space

        This ensures that insignificant whitespace changes don't trigger
        unnecessary re-embeddings.
        """
        normalised = re.sub(r"\s+", " ", text.strip())
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()
