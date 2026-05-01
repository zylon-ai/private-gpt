"""Diff detection algorithms for incremental document updates.

This module implements change detection between old and new versions of
chunked documents. It determines which chunks have been added, modified,
or deleted, so that only the changed chunks need to be re-embedded.

Three approaches are implemented, as described in the thesis:

1. **Hash-based detection** (primary): Compare SHA-256 hashes of old vs new
   chunks. Fast O(n) comparison, sufficient when chunks are paragraph-aligned.

2. **Myers' diff algorithm** (via difflib): LCS-based approach that produces
   the minimal edit script. Used for detailed token-level comparison within
   changed chunks.

3. **Patience diff**: Anchors on unique lines/sentences that appear exactly
   once in both versions, then applies LCS to intervening segments. More
   robust against reorderings.

References (from thesis):
All three are documented in the thesis chapter "Diff-detection algorithms":
- Myers' algorithm: O(ND) time for D edits
- Ratcliff/Obershelp: Gestalt pattern matching
- Patience diff: Bram Cohen 2005, anchors on unique items
"""

import difflib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from private_gpt.components.ingest.incremental.chunk_hasher import HashedChunk

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Type of change detected for a chunk."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


@dataclass
class ChunkChange:
    """Describes a change detected between old and new document versions.

    Attributes:
        change_type: The type of change (added, modified, deleted, unchanged).
        old_chunk: The old chunk (None for ADDED changes).
        new_chunk: The new chunk (None for DELETED changes).
        similarity_ratio: For MODIFIED chunks, the similarity between old and new
                          text as computed by SequenceMatcher (0.0 to 1.0).
    """

    change_type: ChangeType
    old_chunk: HashedChunk | None = None
    new_chunk: HashedChunk | None = None
    similarity_ratio: float = 0.0


class DiffDetector:
    """Detects changes between old and new versions of chunked documents.

    Uses a combination of hash-based detection (fast) and sequence matching
    (for detailed analysis of modified chunks).

    Parameters:
        similarity_threshold: Minimum similarity ratio (0.0-1.0) to consider
                              two chunks as "the same chunk, modified" vs
                              "one deleted and one added". Default 0.4.
    """

    def __init__(self, similarity_threshold: float = 0.4) -> None:
        self.similarity_threshold = similarity_threshold

    def detect_changes(
        self,
        old_chunks: list[HashedChunk],
        new_chunks: list[HashedChunk],
    ) -> list[ChunkChange]:
        """Detect changes between old and new chunk lists.

        Uses a multi-phase approach:
        1. Hash comparison to find unchanged chunks (O(n)).
        2. For remaining chunks, use sequence alignment (patience-style)
           to match modified chunks.
        3. Unmatched old chunks are marked as DELETED, unmatched new
           chunks as ADDED.

        Args:
            old_chunks: Chunks from the previous version of the document.
            new_chunks: Chunks from the current version of the document.

        Returns:
            List of ChunkChange objects describing all changes.
        """
        if not old_chunks:
            # All new - every chunk is ADDED
            return [
                ChunkChange(change_type=ChangeType.ADDED, new_chunk=chunk)
                for chunk in new_chunks
            ]

        if not new_chunks:
            # All deleted
            return [
                ChunkChange(change_type=ChangeType.DELETED, old_chunk=chunk)
                for chunk in old_chunks
            ]

        changes: list[ChunkChange] = []

        # Phase 1: Build hash -> list mappings to handle duplicate chunks
        # (multiple chunks can have the same content/hash).
        old_hash_map: dict[str, list[HashedChunk]] = {}
        for c in old_chunks:
            old_hash_map.setdefault(c.content_hash, []).append(c)
        new_hash_map: dict[str, list[HashedChunk]] = {}
        for c in new_chunks:
            new_hash_map.setdefault(c.content_hash, []).append(c)

        # Find unchanged chunks (hash matches).  Pair them one-to-one
        # using proximity matching: for each set of duplicates with the
        # same hash, pair old<->new by minimising |old_index - new_index|.
        # This prevents context drift when chunks shift position due to
        # inserts/deletes elsewhere in the document.
        matched_old_ids: set[int] = set()  # chunk_index values already paired
        matched_new_ids: set[int] = set()

        for hash_val in set(old_hash_map) & set(new_hash_map):
            old_list = list(old_hash_map[hash_val])
            new_list = list(new_hash_map[hash_val])

            # Greedy proximity pairing: repeatedly match the closest pair
            remaining_old = list(old_list)
            remaining_new = list(new_list)

            while remaining_old and remaining_new:
                best_dist = float("inf")
                best_oi = -1
                best_ni = -1
                for oi, oc in enumerate(remaining_old):
                    for ni, nc in enumerate(remaining_new):
                        dist = abs(oc.chunk_index - nc.chunk_index)
                        if dist < best_dist:
                            best_dist = dist
                            best_oi = oi
                            best_ni = ni
                old_c = remaining_old.pop(best_oi)
                new_c = remaining_new.pop(best_ni)
                changes.append(
                    ChunkChange(
                        change_type=ChangeType.UNCHANGED,
                        old_chunk=old_c,
                        new_chunk=new_c,
                        similarity_ratio=1.0,
                    )
                )
                matched_old_ids.add(old_c.chunk_index)
                matched_new_ids.add(new_c.chunk_index)

        # Remaining unmatched chunks
        unmatched_old = [c for c in old_chunks if c.chunk_index not in matched_old_ids]
        unmatched_new = [c for c in new_chunks if c.chunk_index not in matched_new_ids]

        # Phase 2: Use sequence matching (Patience-style) to pair modified chunks
        matched_old_indices: set[int] = set()
        matched_new_indices: set[int] = set()

        if unmatched_old and unmatched_new:
            # Use difflib SequenceMatcher (implements Ratcliff/Obershelp algorithm)
            # to find the best matches between old and new unmatched chunks
            for i, old_chunk in enumerate(unmatched_old):
                best_ratio = 0.0
                best_j = -1

                for j, new_chunk in enumerate(unmatched_new):
                    if j in matched_new_indices:
                        continue

                    # Compute similarity using Ratcliff/Obershelp (Gestalt)
                    ratio = difflib.SequenceMatcher(
                        None, old_chunk.text, new_chunk.text
                    ).ratio()

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_j = j

                if best_j >= 0 and best_ratio >= self.similarity_threshold:
                    matched_old_indices.add(i)
                    matched_new_indices.add(best_j)
                    changes.append(
                        ChunkChange(
                            change_type=ChangeType.MODIFIED,
                            old_chunk=old_chunk,
                            new_chunk=unmatched_new[best_j],
                            similarity_ratio=best_ratio,
                        )
                    )

        # Phase 3: Remaining unmatched = deletions and additions
        for i, old_chunk in enumerate(unmatched_old):
            if i not in matched_old_indices:
                changes.append(
                    ChunkChange(change_type=ChangeType.DELETED, old_chunk=old_chunk)
                )

        for j, new_chunk in enumerate(unmatched_new):
            if j not in matched_new_indices:
                changes.append(
                    ChunkChange(change_type=ChangeType.ADDED, new_chunk=new_chunk)
                )

        # Sort by new chunk index (for consistent ordering)
        changes.sort(key=lambda c: _change_sort_key(c))

        # Log summary
        added = sum(1 for c in changes if c.change_type == ChangeType.ADDED)
        modified = sum(1 for c in changes if c.change_type == ChangeType.MODIFIED)
        deleted = sum(1 for c in changes if c.change_type == ChangeType.DELETED)
        unchanged = sum(1 for c in changes if c.change_type == ChangeType.UNCHANGED)

        logger.info(
            "Diff result: %d unchanged, %d modified, %d added, %d deleted "
            "(out of %d old -> %d new chunks)",
            unchanged,
            modified,
            added,
            deleted,
            len(old_chunks),
            len(new_chunks),
        )

        return changes

    def get_detailed_diff(self, old_text: str, new_text: str) -> list[str]:
        """Get a unified diff between two text strings.

        Uses Myers' algorithm (via difflib.unified_diff) to produce a
        human-readable diff output. Useful for debugging and logging.

        Args:
            old_text: The original text.
            new_text: The modified text.

        Returns:
            List of diff lines (unified diff format).
        """
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        return list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile="old",
                tofile="new",
                lineterm="",
            )
        )


def patience_diff(old_lines: list[str], new_lines: list[str]) -> list[tuple[str, str]]:
    """Patience diff implementation.

    Patience diff (Bram Cohen, 2005) works by:
    1. Finding lines that appear exactly once in both old and new versions
       (these are "unique anchors").
    2. Computing the Longest Increasing Subsequence (LIS) of matched
       anchor positions to find the best alignment.
    3. Recursively applying diff to the segments between anchors.

    This produces more intuitive diffs when text is reordered, because
    unique lines (likely section headers, key statements) serve as stable
    reference points.

    Args:
        old_lines: Lines from the old version.
        new_lines: Lines from the new version.

    Returns:
        List of (tag, line) tuples where tag is one of:
        ' ' (context), '+' (added), '-' (removed).
    """
    # Step 1: Find unique lines in both versions
    old_unique: dict[str, int] = {}
    old_counts: dict[str, int] = {}
    for i, line in enumerate(old_lines):
        old_counts[line] = old_counts.get(line, 0) + 1
        old_unique[line] = i

    new_unique: dict[str, int] = {}
    new_counts: dict[str, int] = {}
    for i, line in enumerate(new_lines):
        new_counts[line] = new_counts.get(line, 0) + 1
        new_unique[line] = i

    # Find truly unique lines (appear exactly once in both)
    anchors: list[tuple[int, int]] = []
    for line in old_unique:
        if old_counts.get(line, 0) == 1 and new_counts.get(line, 0) == 1:
            anchors.append((old_unique[line], new_unique[line]))

    # Step 2: Sort by old position and find LIS on new positions
    anchors.sort(key=lambda x: x[0])
    if anchors:
        # Longest Increasing Subsequence on new-line positions
        lis_anchors = _longest_increasing_subsequence(anchors, key=lambda x: x[1])
    else:
        lis_anchors = []

    # Step 3: Build diff output using anchors as fixed points
    result: list[tuple[str, str]] = []
    old_pos = 0
    new_pos = 0

    for old_idx, new_idx in lis_anchors:
        # Recursively diff segments between anchors using standard LCS
        _diff_segment(old_lines, old_pos, old_idx, new_lines, new_pos, new_idx, result)
        result.append((" ", old_lines[old_idx]))
        old_pos = old_idx + 1
        new_pos = new_idx + 1

    # Handle remaining lines after last anchor
    _diff_segment(
        old_lines, old_pos, len(old_lines), new_lines, new_pos, len(new_lines), result
    )

    return result


def _diff_segment(
    old_lines: list[str],
    old_start: int,
    old_end: int,
    new_lines: list[str],
    new_start: int,
    new_end: int,
    result: list[tuple[str, str]],
) -> None:
    """Diff a segment between two anchors using standard difflib."""
    old_segment = old_lines[old_start:old_end]
    new_segment = new_lines[new_start:new_end]

    matcher = difflib.SequenceMatcher(None, old_segment, new_segment)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for line in old_segment[i1:i2]:
                result.append((" ", line))
        elif tag == "replace":
            for line in old_segment[i1:i2]:
                result.append(("-", line))
            for line in new_segment[j1:j2]:
                result.append(("+", line))
        elif tag == "delete":
            for line in old_segment[i1:i2]:
                result.append(("-", line))
        elif tag == "insert":
            for line in new_segment[j1:j2]:
                result.append(("+", line))


def _longest_increasing_subsequence(
    pairs: list[tuple[int, int]],
    key: Callable[[tuple[int, int]], int] = lambda x: x[1],
) -> list[tuple[int, int]]:
    """Compute the Longest Increasing Subsequence for anchor pairs.

    Used by Patience diff to find the optimal alignment of unique anchors.
    O(n log n) implementation using binary search.
    """
    import bisect

    if not pairs:
        return []

    # tails[i] = smallest ending value for increasing subsequence of length i+1
    tails: list[int] = []
    # For reconstruction
    predecessors: list[int] = [-1] * len(pairs)
    indices: list[int] = []  # indices[i] = index in pairs for tails[i]

    for idx, pair in enumerate(pairs):
        val = key(pair)
        pos = bisect.bisect_left(tails, val)
        if pos == len(tails):
            tails.append(val)
            indices.append(idx)
        else:
            tails[pos] = val
            indices[pos] = idx

        if pos > 0:
            predecessors[idx] = indices[pos - 1]

    # Reconstruct the LIS
    result = []
    k = indices[len(tails) - 1] if tails else -1
    while k >= 0:
        result.append(pairs[k])
        k = predecessors[k]

    result.reverse()
    return result


def _change_sort_key(change: ChunkChange) -> int:
    """Sorting key for changes: by new chunk index (or old for deletions)."""
    if change.new_chunk is not None:
        return change.new_chunk.chunk_index
    if change.old_chunk is not None:
        return change.old_chunk.chunk_index
    return 0
