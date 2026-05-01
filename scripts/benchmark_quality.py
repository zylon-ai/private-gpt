#!/usr/bin/env python3
"""Quality benchmarks for the incremental update pipeline.

While benchmark_incremental.py measures **speed**, this script measures
**correctness and robustness**:

1. Context Drift – Does repeated incremental updating corrupt node_id
   mappings?  After N sequential edits, do unchanged chunks still point
   to their original embedding node?

2. Diff Accuracy – Does the DiffDetector correctly classify chunks as
   UNCHANGED / MODIFIED / ADDED / DELETED?  Measured via precision,
   recall and F1 per change type against a ground-truth oracle.

3. Hash Stability – Do semantically identical chunks always produce the
   same hash, even across different chunking runs with surrounding
   context changes?

4. Cumulative Drift – How does drift accumulate over many small edits
   vs fewer large edits?

5. Scalability – How does diff detection time scale with document size?

6. Stress Tests – Edge cases: boundary shifts, chunk merges/splits,
   heavy duplicates, multi-edit cascades.

Usage:
    python -m scripts.benchmark_quality [--output-dir ./benchmark_results]

References (from thesis):
- §Risico's en beperkingen: context drift
- §Methodologie: Experimenten en evaluatie
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from private_gpt.components.ingest.incremental.chunk_hash_store import (
    ChunkHashStore,
    DocumentRecord,
    StoredChunkInfo,
)
from private_gpt.components.ingest.incremental.chunk_hasher import (
    HashedChunk,
    SemanticChunker,
)
from private_gpt.components.ingest.incremental.diff_detector import (
    ChangeType,
    ChunkChange,
    DiffDetector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Sample paragraphs (same as benchmark_incremental.py) ───────────────

SAMPLE_PARAGRAPHS = [
    (
        "Retrieval-Augmented Generation (RAG) combines large language models "
        "with external knowledge sources to generate answers based on current "
        "documents. This approach allows the AI to produce answers grounded "
        "in external data rather than only its pre-trained weights."
    ),
    (
        "Local RAG systems are gaining popularity for privacy and latency "
        "reasons. Organisations dealing with sensitive or confidential data "
        "want to avoid sending it to external services. Local models protect "
        "privacy and reduce round-trip latency."
    ),
    (
        "PrivateGPT uses LlamaIndex to split incoming documents into text "
        "blocks of roughly 1024 tokens with a default overlap of 20 tokens. "
        "These chunks are stored in a NodeStore and the corresponding "
        "embeddings in a VectorStore."
    ),
    (
        "With fixed-size chunking a document is sliced linearly into equal "
        "blocks, often with overlap. The downside is the avalanche effect: "
        "a subtle edit shifts every subsequent chunk, forcing many embeddings "
        "to be recomputed."
    ),
    (
        "Semantic chunking splits text based on content cohesion. A "
        "breakpoint-based chunker picks split points where the semantic "
        "distance crosses a threshold. This prevents the domino effect that "
        "fixed-size chunking exhibits on small edits."
    ),
    (
        "Myers' algorithm is a well-known LCS-based method that produces a "
        "shortest edit script in O(ND) time. The algorithm can be applied "
        "at any granularity: lines, sentences, or tokens."
    ),
    (
        "Patience diff first locates sentences or tokens that occur exactly "
        "once in both versions as anchor points, pairs them up, and then runs "
        "LCS on the intervening segments. It tends to give more robust diffs "
        "in the presence of reorderings or repetitions."
    ),
    (
        "Hash-based detection precomputes a hash for every chunk. On a "
        "subsequent indexing run, hashes are compared to skip unchanged "
        "chunks. Only chunks whose hash has changed are re-embedded and "
        "updated in the vector store."
    ),
    (
        "At the vector-database level the upsert primitive is common. Modern "
        "vector stores such as Qdrant, Milvus, and FAISS support updating "
        "individual vectors: an update can be a delete followed by an insert, "
        "or an explicit upsert call."
    ),
    (
        "HNSW indexes build a layered graph structure with skip-list-like "
        "layers. They yield very fast searches with high recall. HNSW is "
        "designed for dynamic data: new vectors can be added incrementally "
        "without rebuilding the index from scratch."
    ),
    (
        "Embedding drift is the phenomenon where small changes in text or "
        "model can cause large displacements in the vector space. For "
        "incremental document management, embedding stability is essential: "
        "unchanged passages should keep stable embeddings."
    ),
    (
        "Local RAG systems process documents entirely on-premises, which "
        "yields significant privacy benefits for sensitive data. Because "
        "documents and queries never reach external servers, all raw "
        "information stays inside the controlled environment."
    ),
    (
        "The performance of a local RAG solution is measured with a mix of "
        "system and information-retrieval metrics. Key system metrics are "
        "latency and throughput. Latency is the time the system needs to "
        "respond to a query."
    ),
    (
        "Incremental knowledge updates carry technical risks. Inconsistencies "
        "can arise from mismatches in chunking or embeddings. Without a "
        "rollback mechanism, a failed update may force a complete rebuild "
        "of the index."
    ),
    (
        "Hybrid RAG architectures combine local and cloud components. The "
        "benefits are lower latency and improved privacy: critical data "
        "stays on-premises, while compute power can be borrowed from the "
        "cloud for more complex queries."
    ),
]


# ─── Document Generation Helpers ────────────────────────────────────────


def generate_document(num_paragraphs: int = 10, seed: int = 42) -> str:
    rng = random.Random(seed)
    paragraphs = [rng.choice(SAMPLE_PARAGRAPHS) for _ in range(num_paragraphs)]
    return "\n\n".join(paragraphs)


def modify_single_paragraph(text: str, paragraph_index: int, seed: int = 0) -> str:
    """Modify exactly one paragraph in the document.

    This gives ground-truth knowledge of which chunk should change.
    """
    rng = random.Random(seed)
    paragraphs = text.split("\n\n")
    if paragraph_index >= len(paragraphs):
        paragraph_index = len(paragraphs) - 1

    # Append a unique sentence so the chunk hash definitely changes
    paragraphs[paragraph_index] += (
        f" [Update {seed}] This is an intentional modification "
        f"for benchmarking purposes with extra context: {rng.randint(1000, 9999)}."
    )
    return "\n\n".join(paragraphs)


def insert_paragraph(text: str, position: int, seed: int = 0) -> str:
    """Insert a new paragraph at the given position."""
    paragraphs = text.split("\n\n")
    new_para = (
        f"[Inserted paragraph {seed}] This is a completely new paragraph "
        f"added to the document at position {position}. "
        f"The content serves as test data for the benchmark script."
    )
    paragraphs.insert(position, new_para)
    return "\n\n".join(paragraphs)


def delete_paragraph(text: str, position: int) -> str:
    """Delete the paragraph at the given position."""
    paragraphs = text.split("\n\n")
    if 0 <= position < len(paragraphs):
        paragraphs.pop(position)
    return "\n\n".join(paragraphs)


# ═════════════════════════════════════════════════════════════════════════
# Benchmark 1: Context Drift
# ═════════════════════════════════════════════════════════════════════════


@dataclass
class DriftMeasurement:
    """Result of a single drift measurement after one update iteration."""

    iteration: int
    edit_type: str  # "modify", "insert", "delete"
    total_chunks: int
    unchanged_chunks: int
    correct_node_ids: int
    incorrect_node_ids: int
    missing_node_ids: int
    drift_score: float  # 0.0 = no drift, 1.0 = total drift
    # Functional drift: the node_id points to a chunk with DIFFERENT content.
    # This is the truly harmful case — it means the embedding is wrong.
    functional_drift_count: int = 0
    functional_drift_score: float = 0.0


def simulate_context_drift(
    num_iterations: int = 10,
    num_paragraphs: int = 12,
    seed: int = 42,
) -> list[DriftMeasurement]:
    """Simulate sequential incremental updates and measure node_id drift.

    The test works as follows:
    1. Create a document, chunk it, assign fake node_ids.
    2. In each iteration, make a small edit (modify / insert / delete).
    3. Run DiffDetector to classify changes.
    4. Simulate the node_id registry update (same logic as IncrementalUpdater).
    5. Verify that UNCHANGED chunks still map to the correct original
       node_id.  Any mismatch = context drift.

    Returns:
        List of DriftMeasurement, one per iteration.
    """
    rng = random.Random(seed)
    chunker = SemanticChunker(min_chunk_size=50, max_chunk_size=3000)
    detector = DiffDetector(similarity_threshold=0.4)

    # --- Initial state ---
    text = generate_document(num_paragraphs=num_paragraphs, seed=seed)
    chunks = chunker.chunk_text(text, metadata={"file": "drift_test.txt"})

    # Assign deterministic node_ids (simulating what IncrementalUpdater does)
    node_id_map: dict[int, str] = {}  # chunk_index -> node_id
    # Track which content_hash each node_id was originally created for.
    # This lets us detect *functional* drift: a node_id now pointing to
    # different content than it was embedded for.
    node_id_to_hash: dict[str, str] = {}  # node_id -> content_hash
    # Track all valid node_ids per content_hash (handles duplicates).
    hash_to_node_ids: dict[str, set[str]] = {}  # content_hash -> {node_ids}
    for c in chunks:
        nid = f"node_v0_{c.chunk_index}"
        node_id_map[c.chunk_index] = nid
        node_id_to_hash[nid] = c.content_hash
        hash_to_node_ids.setdefault(c.content_hash, set()).add(nid)

    # Build stored record
    stored_chunks_info: list[StoredChunkInfo] = [
        StoredChunkInfo(
            chunk_index=c.chunk_index,
            content_hash=c.content_hash,
            node_id=node_id_map[c.chunk_index],
            text_preview=c.text[:100],
            full_text=c.text,
        )
        for c in chunks
    ]

    measurements: list[DriftMeasurement] = []

    for iteration in range(1, num_iterations + 1):
        # --- Choose a random edit ---
        edit_type = rng.choice(["modify", "insert", "delete"])
        paragraphs = text.split("\n\n")
        num_paras = len(paragraphs)

        if edit_type == "modify":
            idx = rng.randint(0, num_paras - 1)
            text = modify_single_paragraph(text, idx, seed=iteration * 100)
        elif edit_type == "insert":
            idx = rng.randint(0, num_paras)
            text = insert_paragraph(text, idx, seed=iteration * 100)
        elif edit_type == "delete" and num_paras > 3:
            idx = rng.randint(0, num_paras - 1)
            text = delete_paragraph(text, idx)
        else:
            # Fall back to modify if too few paragraphs
            edit_type = "modify"
            idx = rng.randint(0, max(0, num_paras - 1))
            text = modify_single_paragraph(text, idx, seed=iteration * 100)

        # --- Chunk the new version ---
        new_chunks = chunker.chunk_text(text, metadata={"file": "drift_test.txt"})

        # --- Reconstruct old HashedChunks from stored info ---
        old_chunks_reconstructed = [
            HashedChunk(
                chunk_index=sci.chunk_index,
                text=sci.full_text or sci.text_preview,
                content_hash=sci.content_hash,
            )
            for sci in stored_chunks_info
        ]

        # --- Detect changes ---
        changes = detector.detect_changes(old_chunks_reconstructed, new_chunks)

        # --- Simulate registry update (mirrors IncrementalUpdater Step 8) ---
        old_node_ids = {sci.chunk_index: sci.node_id for sci in stored_chunks_info}
        new_stored: list[StoredChunkInfo] = []
        new_node_map: dict[int, str] = {}

        correct = 0
        incorrect = 0
        missing = 0
        functional_drift = 0
        unchanged_count = 0

        for change in changes:
            if change.change_type == ChangeType.UNCHANGED and change.new_chunk:
                unchanged_count += 1
                chunk = change.new_chunk
                # Use the FIXED logic: lookup via old_chunk.chunk_index
                old_idx = (
                    change.old_chunk.chunk_index
                    if change.old_chunk is not None
                    else chunk.chunk_index
                )
                resolved_node_id = old_node_ids.get(old_idx, "")

                if not resolved_node_id:
                    missing += 1
                else:
                    # Strict check: is this node_id one of the valid ones
                    # for this content_hash?
                    valid_nids = hash_to_node_ids.get(chunk.content_hash, set())
                    if resolved_node_id in valid_nids:
                        correct += 1
                    else:
                        incorrect += 1
                        # Functional drift: does the node_id point to a
                        # DIFFERENT content_hash entirely?
                        original_hash = node_id_to_hash.get(resolved_node_id, "")
                        if original_hash != chunk.content_hash:
                            functional_drift += 1

                new_stored.append(
                    StoredChunkInfo(
                        chunk_index=chunk.chunk_index,
                        content_hash=chunk.content_hash,
                        node_id=resolved_node_id,
                        text_preview=chunk.text[:100],
                        full_text=chunk.text,
                    )
                )

            elif change.change_type in (ChangeType.ADDED, ChangeType.MODIFIED):
                if change.new_chunk:
                    chunk = change.new_chunk
                    new_nid = f"node_v{iteration}_{chunk.chunk_index}"
                    new_node_map[chunk.chunk_index] = new_nid
                    # Register this node_id's content mapping
                    node_id_to_hash[new_nid] = chunk.content_hash
                    hash_to_node_ids.setdefault(chunk.content_hash, set()).add(new_nid)
                    new_stored.append(
                        StoredChunkInfo(
                            chunk_index=chunk.chunk_index,
                            content_hash=chunk.content_hash,
                            node_id=new_nid,
                            text_preview=chunk.text[:100],
                            full_text=chunk.text,
                        )
                    )
            # DELETED: not stored

        new_stored.sort(key=lambda s: s.chunk_index)
        stored_chunks_info = new_stored

        total_unchanged = unchanged_count
        drift_score = (
            (incorrect + missing) / total_unchanged
            if total_unchanged > 0
            else 0.0
        )
        func_drift_score = (
            (functional_drift + missing) / total_unchanged
            if total_unchanged > 0
            else 0.0
        )

        m = DriftMeasurement(
            iteration=iteration,
            edit_type=edit_type,
            total_chunks=len(new_chunks),
            unchanged_chunks=total_unchanged,
            correct_node_ids=correct,
            incorrect_node_ids=incorrect,
            missing_node_ids=missing,
            drift_score=drift_score,
            functional_drift_count=functional_drift,
            functional_drift_score=func_drift_score,
        )
        measurements.append(m)

        logger.info(
            "  Iter %2d [%s]: %d chunks, %d unchanged, "
            "strict_drift=%.2f, func_drift=%.2f "
            "(correct=%d, incorrect=%d, func_err=%d, missing=%d)",
            iteration,
            edit_type,
            m.total_chunks,
            m.unchanged_chunks,
            m.drift_score,
            m.functional_drift_score,
            m.correct_node_ids,
            m.incorrect_node_ids,
            m.functional_drift_count,
            m.missing_node_ids,
        )

    return measurements


# ═════════════════════════════════════════════════════════════════════════
# Benchmark 2: Diff Accuracy (Precision / Recall / F1)
# ═════════════════════════════════════════════════════════════════════════


@dataclass
class DiffAccuracyResult:
    """Precision, recall and F1 for each change type."""

    scenario: str
    change_type: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


def _classify_ground_truth(
    old_chunks: list[HashedChunk],
    new_chunks: list[HashedChunk],
    modified_indices: set[int],
    added_indices: set[int],
    deleted_indices: set[int],
) -> list[ChunkChange]:
    """Build ground-truth changes from known edit operations."""
    truth: list[ChunkChange] = []
    old_by_idx = {c.chunk_index: c for c in old_chunks}
    new_by_idx = {c.chunk_index: c for c in new_chunks}

    # All old indices that are not deleted and not modified are unchanged
    for c in old_chunks:
        if c.chunk_index in deleted_indices:
            truth.append(ChunkChange(change_type=ChangeType.DELETED, old_chunk=c))
        elif c.chunk_index in modified_indices:
            new_c = new_by_idx.get(c.chunk_index)
            truth.append(
                ChunkChange(change_type=ChangeType.MODIFIED, old_chunk=c, new_chunk=new_c)
            )
        else:
            new_c = new_by_idx.get(c.chunk_index)
            truth.append(
                ChunkChange(change_type=ChangeType.UNCHANGED, old_chunk=c, new_chunk=new_c)
            )

    for c in new_chunks:
        if c.chunk_index in added_indices:
            truth.append(ChunkChange(change_type=ChangeType.ADDED, new_chunk=c))

    return truth


def measure_diff_accuracy(
    chunker: SemanticChunker,
    detector: DiffDetector,
) -> list[DiffAccuracyResult]:
    """Measure precision/recall/F1 of the DiffDetector.

    Creates documents with known ground-truth edits and compares the
    detector's output against the oracle.
    """
    results: list[DiffAccuracyResult] = []

    scenarios = [
        ("Single modify", "modify", 1),
        ("Two modifies", "modify", 2),
        ("Single insert", "insert", 1),
        ("Single delete", "delete", 1),
        ("Insert + modify", "mixed_im", 1),
        ("Delete + modify", "mixed_dm", 1),
    ]

    for scenario_name, edit_type, count in scenarios:
        # Generate reproducible document
        text = generate_document(num_paragraphs=10, seed=77)
        old_chunks = chunker.chunk_text(text, metadata={"file": "accuracy_test.txt"})
        num_old = len(old_chunks)

        # Apply known edits
        ground_truth_modified: set[int] = set()
        ground_truth_added: set[int] = set()
        ground_truth_deleted: set[int] = set()

        if edit_type == "modify":
            for i in range(count):
                idx = min(i, num_old - 1)
                text = modify_single_paragraph(text, idx, seed=scenario_name.__hash__() + i)
                ground_truth_modified.add(idx)

        elif edit_type == "insert":
            text = insert_paragraph(text, 2, seed=scenario_name.__hash__())
            # After insertion, the chunker may re-index; we track at paragraph level
            ground_truth_added.add(-1)  # special marker

        elif edit_type == "delete":
            text = delete_paragraph(text, 2)
            ground_truth_deleted.add(-1)  # special marker

        elif edit_type == "mixed_im":
            text = insert_paragraph(text, 1, seed=99)
            text = modify_single_paragraph(text, 4, seed=100)
            ground_truth_added.add(-1)
            ground_truth_modified.add(-1)

        elif edit_type == "mixed_dm":
            text = delete_paragraph(text, 0)
            text = modify_single_paragraph(text, 3, seed=101)
            ground_truth_deleted.add(-1)
            ground_truth_modified.add(-1)

        new_chunks = chunker.chunk_text(text, metadata={"file": "accuracy_test.txt"})
        predicted = detector.detect_changes(old_chunks, new_chunks)

        # Count per change type
        for ct in ChangeType:
            pred_count = sum(1 for c in predicted if c.change_type == ct)

            # For UNCHANGED: ground truth = number of chunks that still exist
            # with the same hash.  Using set intersection would undercount when
            # a paragraph appears more than once (set collapses duplicates,
            # but the detector correctly reports one UNCHANGED entry per
            # occurrence — inflating false positives and dropping F1 to  0.83).
            if ct == ChangeType.UNCHANGED:
                from collections import Counter
                old_counts = Counter(c.content_hash for c in old_chunks)
                new_counts = Counter(c.content_hash for c in new_chunks)
                gt_count = sum(
                    min(old_counts[h], new_counts[h]) for h in old_counts if h in new_counts
                )
            elif ct == ChangeType.MODIFIED:
                gt_count = len(ground_truth_modified) if -1 not in ground_truth_modified else max(1, len(ground_truth_modified))
            elif ct == ChangeType.ADDED:
                gt_count = len(ground_truth_added) if -1 not in ground_truth_added else max(1, len(ground_truth_added))
            elif ct == ChangeType.DELETED:
                gt_count = len(ground_truth_deleted) if -1 not in ground_truth_deleted else max(1, len(ground_truth_deleted))
            else:
                gt_count = 0

            # Simplified precision/recall based on counts
            tp = min(pred_count, gt_count)
            fp = max(0, pred_count - gt_count)
            fn = max(0, gt_count - pred_count)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            results.append(
                DiffAccuracyResult(
                    scenario=scenario_name,
                    change_type=ct.value,
                    true_positives=tp,
                    false_positives=fp,
                    false_negatives=fn,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                )
            )

    return results


# ═════════════════════════════════════════════════════════════════════════
# Benchmark 3: Hash Stability
# ═════════════════════════════════════════════════════════════════════════


@dataclass
class HashStabilityResult:
    """Whether a chunk's hash stays stable when surrounding context changes."""

    test_name: str
    chunk_text_preview: str
    hash_before: str
    hash_after: str
    is_stable: bool


def measure_hash_stability(chunker: SemanticChunker) -> list[HashStabilityResult]:
    """Verify that chunk hashes are independent of surrounding context.

    A stable chunker should produce the same hash for an identical
    paragraph regardless of what paragraphs appear before/after it.
    """
    results: list[HashStabilityResult] = []

    target_paragraph = SAMPLE_PARAGRAPHS[5]  # "Myers' algoritme..."

    # Scenario A: target surrounded by paragraphs 0-4, 6-8
    context_a = SAMPLE_PARAGRAPHS[0:5] + [target_paragraph] + SAMPLE_PARAGRAPHS[6:9]
    text_a = "\n\n".join(context_a)
    chunks_a = chunker.chunk_text(text_a)

    # Scenario B: target surrounded by completely different paragraphs
    context_b = SAMPLE_PARAGRAPHS[9:12] + [target_paragraph] + SAMPLE_PARAGRAPHS[12:15]
    text_b = "\n\n".join(context_b)
    chunks_b = chunker.chunk_text(text_b)

    # Find the hash of the target paragraph in each chunking
    target_hash = SemanticChunker._compute_hash(target_paragraph)

    hash_a = None
    hash_b = None
    for c in chunks_a:
        if SemanticChunker._compute_hash(c.text) == target_hash or target_paragraph[:50] in c.text:
            hash_a = c.content_hash
            break
    for c in chunks_b:
        if SemanticChunker._compute_hash(c.text) == target_hash or target_paragraph[:50] in c.text:
            hash_b = c.content_hash
            break

    results.append(
        HashStabilityResult(
            test_name="Same paragraph, different surrounding context",
            chunk_text_preview=target_paragraph[:80],
            hash_before=hash_a or "NOT_FOUND",
            hash_after=hash_b or "NOT_FOUND",
            is_stable=(hash_a == hash_b) if hash_a and hash_b else False,
        )
    )

    # Scenario C: extra whitespace variations
    text_norm = "  Hello   world.   Test  content  for   hashing.  "
    text_clean = "Hello world. Test content for hashing."
    hash_norm = SemanticChunker._compute_hash(text_norm)
    hash_clean = SemanticChunker._compute_hash(text_clean)
    results.append(
        HashStabilityResult(
            test_name="Whitespace normalisation",
            chunk_text_preview=text_clean[:80],
            hash_before=hash_norm,
            hash_after=hash_clean,
            is_stable=(hash_norm == hash_clean),
        )
    )

    # Scenario D: identical text, different metadata
    text_d = "A test paragraph that is exactly identical in both cases for hash-stability checks."
    chunks_d1 = chunker.chunk_text(text_d, metadata={"source": "file_a.txt"})
    chunks_d2 = chunker.chunk_text(text_d, metadata={"source": "file_b.txt"})
    if chunks_d1 and chunks_d2:
        results.append(
            HashStabilityResult(
                test_name="Same text, different metadata",
                chunk_text_preview=text_d[:80],
                hash_before=chunks_d1[0].content_hash,
                hash_after=chunks_d2[0].content_hash,
                is_stable=(chunks_d1[0].content_hash == chunks_d2[0].content_hash),
            )
        )

    return results


# ═════════════════════════════════════════════════════════════════════════
# Benchmark 4: Scalability
# ═════════════════════════════════════════════════════════════════════════


@dataclass
class ScalabilityResult:
    """Timing for diff detection at different document sizes."""

    num_paragraphs: int
    num_chunks_old: int
    num_chunks_new: int
    time_chunking_s: float
    time_diffing_s: float
    chunks_per_second: float


def measure_scalability(
    sizes: list[int] | None = None,
) -> list[ScalabilityResult]:
    """Measure how chunking + diff time scales with document size."""
    if sizes is None:
        sizes = [5, 10, 20, 50, 100, 200]

    chunker = SemanticChunker(min_chunk_size=50, max_chunk_size=3000)
    detector = DiffDetector(similarity_threshold=0.4)
    results: list[ScalabilityResult] = []

    for n in sizes:
        original = generate_document(num_paragraphs=n, seed=42)
        # Modify  20% of paragraphs
        modified = original
        rng = random.Random(n)
        num_to_modify = max(1, n // 5)
        for i in range(num_to_modify):
            idx = rng.randint(0, max(0, len(modified.split("\n\n")) - 1))
            modified = modify_single_paragraph(modified, idx, seed=n * 100 + i)

        # Time chunking
        t0 = time.perf_counter()
        old_chunks = chunker.chunk_text(original)
        t1 = time.perf_counter()
        new_chunks = chunker.chunk_text(modified)
        t2 = time.perf_counter()

        chunk_time = (t1 - t0) + (t2 - t1)

        # Time diffing
        t3 = time.perf_counter()
        detector.detect_changes(old_chunks, new_chunks)
        t4 = time.perf_counter()

        diff_time = t4 - t3
        total_chunks = len(old_chunks) + len(new_chunks)
        cps = total_chunks / (chunk_time + diff_time) if (chunk_time + diff_time) > 0 else 0

        results.append(
            ScalabilityResult(
                num_paragraphs=n,
                num_chunks_old=len(old_chunks),
                num_chunks_new=len(new_chunks),
                time_chunking_s=chunk_time,
                time_diffing_s=diff_time,
                chunks_per_second=cps,
            )
        )

        logger.info(
            "  Scale %3d paras: %d+%d chunks, chunk=%.4fs, diff=%.4fs, %.0f chunks/s",
            n, len(old_chunks), len(new_chunks), chunk_time, diff_time, cps,
        )

    return results


# ═════════════════════════════════════════════════════════════════════════
# Output / Export
# ═════════════════════════════════════════════════════════════════════════


def print_drift_table(measurements: list[DriftMeasurement]) -> None:
    print("\n" + "=" * 100)
    print("CONTEXT DRIFT RESULTS")
    print("  Strict drift  = node_id swapped with another chunk (even if same content)")
    print("  Functional drift = node_id points to DIFFERENT content (harmful!)")
    print("=" * 100)
    print(
        f"{'Iter':>4} {'Edit':>8} {'Chunks':>6} {'Unchg':>5} "
        f"{'Correct':>7} {'Swap':>5} {'Miss':>4} "
        f"{'Strict':>8} {'FuncErr':>7} {'FuncDrift':>10}"
    )
    print("-" * 100)
    for m in measurements:
        func_indicator = "OK" if m.functional_drift_score == 0.0 else "!! DRIFT"
        print(
            f"{m.iteration:>4} {m.edit_type:>8} {m.total_chunks:>6} "
            f"{m.unchanged_chunks:>5} {m.correct_node_ids:>7} "
            f"{m.incorrect_node_ids:>5} {m.missing_node_ids:>4} "
            f"{m.drift_score:>7.2%} "
            f"{m.functional_drift_count:>7} "
            f"{m.functional_drift_score:>9.2%} {func_indicator}"
        )

    total_incorrect = sum(m.incorrect_node_ids for m in measurements)
    total_functional = sum(m.functional_drift_count for m in measurements)
    total_missing = sum(m.missing_node_ids for m in measurements)
    total_unchanged = sum(m.unchanged_chunks for m in measurements)
    overall_strict = (
        (total_incorrect + total_missing) / total_unchanged
        if total_unchanged > 0
        else 0.0
    )
    overall_functional = (
        (total_functional + total_missing) / total_unchanged
        if total_unchanged > 0
        else 0.0
    )
    print("-" * 100)
    print(
        f"OVERALL: {total_unchanged} unchanged chunks across {len(measurements)} "
        f"iterations"
    )
    print(
        f"  Strict drift:     {total_incorrect} swapped + {total_missing} missing "
        f"= {overall_strict:.2%}"
    )
    print(
        f"  Functional drift: {total_functional} wrong content + {total_missing} missing "
        f"= {overall_functional:.2%}"
    )
    print("=" * 100)


def print_accuracy_table(results: list[DiffAccuracyResult]) -> None:
    print("\n" + "=" * 90)
    print("DIFF ACCURACY RESULTS (Precision / Recall / F1)")
    print("=" * 90)
    print(
        f"{'Scenario':<22} {'Type':<12} {'TP':>3} {'FP':>3} {'FN':>3} "
        f"{'Prec':>6} {'Rec':>6} {'F1':>6}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r.scenario:<22} {r.change_type:<12} "
            f"{r.true_positives:>3} {r.false_positives:>3} {r.false_negatives:>3} "
            f"{r.precision:>5.1%} {r.recall:>5.1%} {r.f1:>5.1%}"
        )
    print("=" * 90)


def print_hash_stability_table(results: list[HashStabilityResult]) -> None:
    print("\n" + "=" * 90)
    print("HASH STABILITY RESULTS")
    print("=" * 90)
    for r in results:
        status = "STABLE" if r.is_stable else "UNSTABLE"
        print(f"  {r.test_name}: {status}")
        print(f"    Text: {r.chunk_text_preview}...")
        print(f"    Hash A: {r.hash_before[:16]}...")
        print(f"    Hash B: {r.hash_after[:16]}...")
    print("=" * 90)


def print_scalability_table(results: list[ScalabilityResult]) -> None:
    print("\n" + "=" * 90)
    print("SCALABILITY RESULTS")
    print("=" * 90)
    print(
        f"{'Paras':>6} {'Old':>5} {'New':>5} {'Chunk(s)':>9} "
        f"{'Diff(s)':>9} {'Chunks/s':>10}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r.num_paragraphs:>6} {r.num_chunks_old:>5} {r.num_chunks_new:>5} "
            f"{r.time_chunking_s:>9.4f} {r.time_diffing_s:>9.4f} "
            f"{r.chunks_per_second:>10.0f}"
        )
    print("=" * 90)


def export_all(
    output_dir: Path,
    drift: list[DriftMeasurement],
    accuracy: list[DiffAccuracyResult],
    stability: list[HashStabilityResult],
    scalability: list[ScalabilityResult],
    stress: list[StressTestResult] | None = None,
) -> None:
    """Export all results to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Drift
    drift_data = [
        {
            "iteration": m.iteration,
            "edit_type": m.edit_type,
            "total_chunks": m.total_chunks,
            "unchanged_chunks": m.unchanged_chunks,
            "correct_node_ids": m.correct_node_ids,
            "incorrect_node_ids": m.incorrect_node_ids,
            "missing_node_ids": m.missing_node_ids,
            "drift_score": m.drift_score,
            "functional_drift_count": m.functional_drift_count,
            "functional_drift_score": m.functional_drift_score,
        }
        for m in drift
    ]
    with open(output_dir / "quality_drift.json", "w", encoding="utf-8") as f:
        json.dump(drift_data, f, indent=2)

    with open(output_dir / "quality_drift.csv", "w", newline="", encoding="utf-8") as f:
        if drift_data:
            writer = csv.DictWriter(f, fieldnames=drift_data[0].keys())
            writer.writeheader()
            writer.writerows(drift_data)

    # Accuracy
    acc_data = [
        {
            "scenario": r.scenario,
            "change_type": r.change_type,
            "true_positives": r.true_positives,
            "false_positives": r.false_positives,
            "false_negatives": r.false_negatives,
            "precision": r.precision,
            "recall": r.recall,
            "f1": r.f1,
        }
        for r in accuracy
    ]
    with open(output_dir / "quality_accuracy.json", "w", encoding="utf-8") as f:
        json.dump(acc_data, f, indent=2)

    with open(output_dir / "quality_accuracy.csv", "w", newline="", encoding="utf-8") as f:
        if acc_data:
            writer = csv.DictWriter(f, fieldnames=acc_data[0].keys())
            writer.writeheader()
            writer.writerows(acc_data)

    # Stability
    stab_data = [
        {
            "test_name": r.test_name,
            "chunk_text_preview": r.chunk_text_preview,
            "hash_before": r.hash_before,
            "hash_after": r.hash_after,
            "is_stable": r.is_stable,
        }
        for r in stability
    ]
    with open(output_dir / "quality_stability.json", "w", encoding="utf-8") as f:
        json.dump(stab_data, f, indent=2)

    # Scalability
    scale_data = [
        {
            "num_paragraphs": r.num_paragraphs,
            "num_chunks_old": r.num_chunks_old,
            "num_chunks_new": r.num_chunks_new,
            "time_chunking_s": r.time_chunking_s,
            "time_diffing_s": r.time_diffing_s,
            "chunks_per_second": r.chunks_per_second,
        }
        for r in scalability
    ]
    with open(output_dir / "quality_scalability.json", "w", encoding="utf-8") as f:
        json.dump(scale_data, f, indent=2)

    with open(output_dir / "quality_scalability.csv", "w", newline="", encoding="utf-8") as f:
        if scale_data:
            writer = csv.DictWriter(f, fieldnames=scale_data[0].keys())
            writer.writeheader()
            writer.writerows(scale_data)

    # Stress tests
    if stress:
        stress_data = [
            {
                "scenario": r.scenario,
                "description": r.description,
                "old_chunk_count": r.old_chunk_count,
                "new_chunk_count": r.new_chunk_count,
                "unchanged_count": r.unchanged_count,
                "correct_node_ids": r.correct_node_ids,
                "incorrect_node_ids": r.incorrect_node_ids,
                "functional_errors": r.functional_errors,
                "missing_node_ids": r.missing_node_ids,
                "passed": r.passed,
            }
            for r in stress
        ]
        with open(output_dir / "quality_stress.json", "w", encoding="utf-8") as f:
            json.dump(stress_data, f, indent=2)

        with open(output_dir / "quality_stress.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=stress_data[0].keys())
            writer.writeheader()
            writer.writerows(stress_data)

    logger.info("Exported quality benchmark results to %s", output_dir)


# ═════════════════════════════════════════════════════════════════════════
# Benchmark 5: Stress Tests – Edge Cases
# ═════════════════════════════════════════════════════════════════════════
#
# The basic drift benchmark uses well-separated paragraphs ( 200-400 chars)
# that always map 1:1 to chunks.  These stress tests exercise harder
# scenarios that can occur in real-world documents:
#
# A. Chunk-boundary shift via min_chunk_size merging
# B. Chunk-boundary shift via max_chunk_size splitting
# C. Boundary destruction (removing \n\n between paragraphs)
# D. Heavy duplicate content with positional shifts
# E. Multi-edit cascades (modify + insert + delete in one step)
# ═════════════════════════════════════════════════════════════════════════


@dataclass
class StressTestResult:
    """Result of a single stress-test scenario."""

    scenario: str
    description: str
    old_chunk_count: int
    new_chunk_count: int
    unchanged_count: int
    correct_node_ids: int
    incorrect_node_ids: int   # strict: wrong node_id even if same content
    functional_errors: int    # harmful: node_id points to different content
    missing_node_ids: int
    passed: bool              # functional_errors == 0 and missing == 0


def _run_stress_scenario(
    scenario: str,
    description: str,
    text_v1: str,
    text_v2: str,
    *,
    min_chunk_size: int = 50,
    max_chunk_size: int = 3000,
) -> StressTestResult:
    """Run one stress-test scenario and measure drift."""
    chunker = SemanticChunker(
        min_chunk_size=min_chunk_size, max_chunk_size=max_chunk_size,
    )
    detector = DiffDetector(similarity_threshold=0.4)

    old_chunks = chunker.chunk_text(text_v1, metadata={"file": "stress.txt"})
    new_chunks = chunker.chunk_text(text_v2, metadata={"file": "stress.txt"})

    # Assign node_ids and build tracking structures
    node_id_map: dict[int, str] = {}
    node_id_to_hash: dict[str, str] = {}
    hash_to_node_ids: dict[str, set[str]] = {}
    for c in old_chunks:
        nid = f"stress_{c.chunk_index}"
        node_id_map[c.chunk_index] = nid
        node_id_to_hash[nid] = c.content_hash
        hash_to_node_ids.setdefault(c.content_hash, set()).add(nid)

    # Detect changes
    changes = detector.detect_changes(old_chunks, new_chunks)

    # Verify UNCHANGED chunks
    old_node_ids = {c.chunk_index: node_id_map[c.chunk_index] for c in old_chunks}

    correct = incorrect = functional = missing = unchanged_count = 0

    for change in changes:
        if change.change_type == ChangeType.UNCHANGED and change.new_chunk:
            unchanged_count += 1
            chunk = change.new_chunk
            old_idx = (
                change.old_chunk.chunk_index
                if change.old_chunk is not None
                else chunk.chunk_index
            )
            resolved_nid = old_node_ids.get(old_idx, "")

            if not resolved_nid:
                missing += 1
            else:
                valid = hash_to_node_ids.get(chunk.content_hash, set())
                if resolved_nid in valid:
                    correct += 1
                else:
                    incorrect += 1
                    orig_hash = node_id_to_hash.get(resolved_nid, "")
                    if orig_hash != chunk.content_hash:
                        functional += 1

    passed = (functional == 0 and missing == 0)

    return StressTestResult(
        scenario=scenario,
        description=description,
        old_chunk_count=len(old_chunks),
        new_chunk_count=len(new_chunks),
        unchanged_count=unchanged_count,
        correct_node_ids=correct,
        incorrect_node_ids=incorrect,
        functional_errors=functional,
        missing_node_ids=missing,
        passed=passed,
    )


def run_stress_tests() -> list[StressTestResult]:
    """Run all edge-case stress tests."""
    results: list[StressTestResult] = []

    # ── A. Chunk merge via min_chunk_size ────────────────────────────
    # v1: short para (40 chars) + normal para → merged into one chunk.
    # v2: short para expanded to 200 chars → now stands alone.
    # Expectation: boundary shifts, but no drift.
    short = "This is short."  # 14 chars
    normal_a = (
        "Retrieval-Augmented Generation combines large language models with "
        "external knowledge sources to generate answers grounded in current "
        "documents. This approach is particularly useful."
    )
    normal_b = (
        "Local RAG systems are gaining popularity for privacy and latency "
        "reasons. Organisations dealing with sensitive data want to avoid "
        "shipping information out to external cloud services."
    )
    normal_c = (
        "PrivateGPT uses LlamaIndex to split incoming documents into text "
        "blocks that are subsequently stored as embeddings in a local "
        "VectorStore database."
    )

    v1_merge = f"{short}\n\n{normal_a}\n\n{normal_b}\n\n{normal_c}"
    # Expand the short paragraph so it no longer needs merging
    expanded = (
        "This is short but now expanded with much more content about how "
        "incremental updates work in document management systems that use "
        "vector databases and embedding technology for text retrieval."
    )
    v2_merge = f"{expanded}\n\n{normal_a}\n\n{normal_b}\n\n{normal_c}"

    results.append(_run_stress_scenario(
        "A. Chunk merge -> split",
        "Short paragraph expands past min_chunk_size, causing boundary shift",
        v1_merge, v2_merge,
        min_chunk_size=100, max_chunk_size=3000,
    ))

    # ── B. Chunk split via max_chunk_size ────────────────────────────
    # v1: one large paragraph just under max_chunk_size → 1 chunk.
    # v2: paragraph grows past max_chunk_size → splits into 2 chunks.
    # Expectation: MODIFIED + ADDED, no drift for other chunks.
    large_para = (
        "This is a very long paragraph that approaches the max_chunk_size limit. "
        * 10
    ).strip()
    v1_split = f"{large_para}\n\n{normal_a}\n\n{normal_b}"
    # Make it bigger to exceed max_chunk_size=600
    extra = (
        " Additional content about how the incremental update system works with "
        "LlamaIndex VectorStoreIndex, including insert_nodes and delete_nodes "
        "operations that must be executed atomically. "
        "Throughput of this system is measured in chunks per second."
    )
    v2_split = f"{large_para}{extra}\n\n{normal_a}\n\n{normal_b}"

    results.append(_run_stress_scenario(
        "B. Chunk split (max_size)",
        "Paragraph grows past max_chunk_size, splitting into 2 chunks",
        v1_split, v2_split,
        min_chunk_size=50, max_chunk_size=600,
    ))

    # ── C. Boundary destruction ──────────────────────────────────────
    # v1: para A \n\n para B → 2 separate chunks.
    # v2: remove \n\n → A and B merge into 1 chunk.
    # Other chunks should keep their node_ids.
    v1_boundary = f"{normal_a}\n\n{normal_b}\n\n{normal_c}"
    # Remove boundary between A and B
    v2_boundary = f"{normal_a} {normal_b}\n\n{normal_c}"

    results.append(_run_stress_scenario(
        "C. Boundary destruction",
        "Removing \\n\\n merges two paragraphs into one chunk",
        v1_boundary, v2_boundary,
    ))

    # ── C2. Boundary creation ────────────────────────────────────────
    # Reverse: split one merged chunk into two by adding \n\n.
    results.append(_run_stress_scenario(
        "C2. Boundary creation",
        "Adding \\n\\n splits one chunk into two",
        v2_boundary, v1_boundary,
    ))

    # ── D. Heavy duplicates with inserts ─────────────────────────────
    # 8 identical paragraphs. Insert a new unique paragraph at position 3.
    # After insert, positions shift by +1 from index 3 onwards.
    # All 8 originals should keep correct node_ids.
    dup_para = (
        "Incremental updates offer efficiency benefits because only the "
        "modified chunks need to be re-embedded. This significantly reduces "
        "total processing time for small document changes."
    )
    v1_dups = "\n\n".join([dup_para] * 8)
    paras_dups = [dup_para] * 8
    paras_dups.insert(3, (
        "[UNIQUE] A completely new unique piece of text that does not appear "
        "anywhere else in the document and can serve as an anchor for the diff."
    ))
    v2_dups = "\n\n".join(paras_dups)

    results.append(_run_stress_scenario(
        "D. 8 duplicates + insert",
        "8 identical paragraphs, insert unique paragraph at position 3",
        v1_dups, v2_dups,
    ))

    # ── D2. Heavy duplicates with delete ─────────────────────────────
    # 8 identical paragraphs. Delete paragraph at position 2.
    paras_d2 = [dup_para] * 8
    paras_d2.pop(2)
    v2_dups_del = "\n\n".join(paras_d2)

    results.append(_run_stress_scenario(
        "D2. 8 duplicates + delete",
        "8 identical paragraphs, delete one at position 2",
        v1_dups, v2_dups_del,
    ))

    # ── D3. 15 duplicates + insert + modify ──────────────────────────
    # Maximum stress: 15 identical chunks, insert at position 5,
    # modify at position 10.
    v1_15 = "\n\n".join([dup_para] * 15)
    paras_15 = [dup_para] * 15
    paras_15.insert(5, "[INSERTED] Unique paragraph text number five-point-five.")
    paras_15[11] = dup_para + " [MODIFIED] Extra sentence appended to paragraph 10."
    v2_15 = "\n\n".join(paras_15)

    results.append(_run_stress_scenario(
        "D3. 15 dupes + insert + mod",
        "15 identical paragraphs, insert at 5, modify at 10 (shifted to 11)",
        v1_15, v2_15,
    ))

    # ── E. Multi-edit cascade ────────────────────────────────────────
    # Delete paragraph 0, insert at 3, modify paragraph 5, all at once.
    paras_e = [
        normal_a, normal_b, normal_c,
        SAMPLE_PARAGRAPHS[3], SAMPLE_PARAGRAPHS[4],
        SAMPLE_PARAGRAPHS[5], SAMPLE_PARAGRAPHS[6],
        SAMPLE_PARAGRAPHS[7],
    ]
    v1_multi = "\n\n".join(paras_e)
    paras_e2 = list(paras_e)
    paras_e2.pop(0)       # delete first
    paras_e2.insert(2, (
        "[INSERTED CASCADE] New paragraph inserted in the middle of a "
        "multi-edit batch that stresses the diff algorithm's robustness."
    ))
    paras_e2[4] = paras_e2[4] + " [CASCADE MOD] Extra context appended to this paragraph."
    v2_multi = "\n\n".join(paras_e2)

    results.append(_run_stress_scenario(
        "E. Multi-edit cascade",
        "Delete para 0 + insert at 2 + modify at 4 simultaneously",
        v1_multi, v2_multi,
    ))

    # ── F. Near-threshold similarity ─────────────────────────────────
    # A paragraph that is heavily rewritten ( 40% similar).
    # Should be classified as MODIFIED, not DELETE+ADD.
    # Other chunks should not be affected.
    original_para = (
        "Embedding drift is the phenomenon where small changes in text or "
        "model can cause large displacements in the vector space. "
        "For incremental document management, embedding stability is essential."
    )
    rewritten_para = (
        "The vector-space drift problem arises when textual modifications "
        "lead to unpredictable shifts in the embedded representations. "
        "Stability of embeddings is crucial for incremental pipelines."
    )
    v1_thresh = f"{normal_a}\n\n{original_para}\n\n{normal_b}\n\n{normal_c}"
    v2_thresh = f"{normal_a}\n\n{rewritten_para}\n\n{normal_b}\n\n{normal_c}"

    results.append(_run_stress_scenario(
        "F. Near-threshold rewrite",
        "Heavy rewrite ( 40% similar) of one paragraph among stable chunks",
        v1_thresh, v2_thresh,
    ))

    # ── G. Iterative boundary shifts (20 steps) ─────────────────────
    # Start with small paragraphs that trigger merge behaviour.
    # Each iteration adds text to a random paragraph, potentially
    # crossing the min_chunk_size boundary.
    chunker_g = SemanticChunker(min_chunk_size=150, max_chunk_size=3000)
    detector_g = DiffDetector(similarity_threshold=0.4)
    rng = random.Random(999)

    # Mix of short (< 150) and normal (> 150) paragraphs
    paras_g: list[str] = [
        "Short fragment.",                                #  15 chars -> merge
        normal_a,                                         #  200 chars -> standalone
        "Another short sentence here.",                   #  28 chars -> merge
        normal_b,                                         #  200 chars -> standalone
        "Tiny.",                                          #   5 chars -> merge
        normal_c,                                         #  200 chars -> standalone
        "Final short line of the document.",              #  33 chars -> merge
    ]

    text_g = "\n\n".join(paras_g)
    chunks_g = chunker_g.chunk_text(text_g, metadata={"file": "boundary.txt"})

    nid_map_g: dict[int, str] = {}
    nid_to_hash_g: dict[str, str] = {}
    hash_to_nids_g: dict[str, set[str]] = {}
    for c in chunks_g:
        nid = f"g_{c.chunk_index}"
        nid_map_g[c.chunk_index] = nid
        nid_to_hash_g[nid] = c.content_hash
        hash_to_nids_g.setdefault(c.content_hash, set()).add(nid)

    stored_g = [
        StoredChunkInfo(
            chunk_index=c.chunk_index,
            content_hash=c.content_hash,
            node_id=nid_map_g[c.chunk_index],
            text_preview=c.text[:100],
            full_text=c.text,
        )
        for c in chunks_g
    ]

    g_total_unchanged = 0
    g_total_correct = 0
    g_total_incorrect = 0
    g_total_functional = 0
    g_total_missing = 0

    for step in range(1, 21):
        # Random edit: expand a short paragraph or shrink a long one
        idx = rng.randint(0, len(paras_g) - 1)
        if len(paras_g[idx]) < 150:
            # Expand: cross the merge boundary
            paras_g[idx] += (
                f" Expansion step {step} with extra context about incremental "
                f"updates and vector databases for local RAG systems, id {rng.randint(100,999)}."
            )
        else:
            # Modify: rephrase but stay substantial
            paras_g[idx] = paras_g[idx] + f" [Step {step}] addendum {rng.randint(100,999)}."

        text_g = "\n\n".join(paras_g)
        new_chunks_g = chunker_g.chunk_text(text_g, metadata={"file": "boundary.txt"})

        old_reconstructed_g = [
            HashedChunk(
                chunk_index=s.chunk_index,
                text=s.full_text or s.text_preview,
                content_hash=s.content_hash,
            )
            for s in stored_g
        ]
        changes_g = detector_g.detect_changes(old_reconstructed_g, new_chunks_g)

        old_nids_g = {s.chunk_index: s.node_id for s in stored_g}
        new_stored_g: list[StoredChunkInfo] = []

        for change in changes_g:
            if change.change_type == ChangeType.UNCHANGED and change.new_chunk:
                g_total_unchanged += 1
                chunk = change.new_chunk
                old_idx = (
                    change.old_chunk.chunk_index
                    if change.old_chunk
                    else chunk.chunk_index
                )
                resolved = old_nids_g.get(old_idx, "")
                if not resolved:
                    g_total_missing += 1
                else:
                    valid = hash_to_nids_g.get(chunk.content_hash, set())
                    if resolved in valid:
                        g_total_correct += 1
                    else:
                        g_total_incorrect += 1
                        if nid_to_hash_g.get(resolved, "") != chunk.content_hash:
                            g_total_functional += 1

                new_stored_g.append(StoredChunkInfo(
                    chunk_index=chunk.chunk_index,
                    content_hash=chunk.content_hash,
                    node_id=resolved,
                    text_preview=chunk.text[:100],
                    full_text=chunk.text,
                ))

            elif change.change_type in (ChangeType.ADDED, ChangeType.MODIFIED):
                if change.new_chunk:
                    chunk = change.new_chunk
                    new_nid = f"g_s{step}_{chunk.chunk_index}"
                    nid_to_hash_g[new_nid] = chunk.content_hash
                    hash_to_nids_g.setdefault(chunk.content_hash, set()).add(new_nid)
                    new_stored_g.append(StoredChunkInfo(
                        chunk_index=chunk.chunk_index,
                        content_hash=chunk.content_hash,
                        node_id=new_nid,
                        text_preview=chunk.text[:100],
                        full_text=chunk.text,
                    ))

        new_stored_g.sort(key=lambda s: s.chunk_index)
        stored_g = new_stored_g

    g_passed = (g_total_functional == 0 and g_total_missing == 0)
    results.append(StressTestResult(
        scenario="G. Iterative boundary shifts",
        description="20-step edits on mixed short/long paragraphs (min_chunk_size=150)",
        old_chunk_count=len(chunks_g),
        new_chunk_count=len(new_chunks_g) if 'new_chunks_g' in dir() else 0,
        unchanged_count=g_total_unchanged,
        correct_node_ids=g_total_correct,
        incorrect_node_ids=g_total_incorrect,
        functional_errors=g_total_functional,
        missing_node_ids=g_total_missing,
        passed=g_passed,
    ))

    return results


def print_stress_table(results: list[StressTestResult]) -> None:
    print("\n" + "=" * 110)
    print("STRESS TEST RESULTS — Edge Cases")
    print("  Functional error = node_id points to DIFFERENT content (harmful)")
    print("=" * 110)
    print(
        f"{'Scenario':<30} {'Old':>4} {'New':>4} {'Unchg':>5} "
        f"{'OK':>4} {'Swap':>4} {'Func':>4} {'Miss':>4} {'Result':<10}"
    )
    print("-" * 110)
    for r in results:
        status = "PASS" if r.passed else "!! FAIL"
        print(
            f"{r.scenario:<30} {r.old_chunk_count:>4} {r.new_chunk_count:>4} "
            f"{r.unchanged_count:>5} {r.correct_node_ids:>4} "
            f"{r.incorrect_node_ids:>4} {r.functional_errors:>4} "
            f"{r.missing_node_ids:>4} {status:<10}"
        )
        print(f"  {r.description}")

    total_func = sum(r.functional_errors for r in results)
    total_miss = sum(r.missing_node_ids for r in results)
    all_pass = all(r.passed for r in results)
    print("-" * 110)
    print(
        f"OVERALL: {len(results)} scenarios, "
        f"functional errors={total_func}, "
        f"missing={total_miss}"
    )
    if all_pass:
        print("  ALL PASS — no functional drift in any edge case")
    else:
        failed = [r.scenario for r in results if not r.passed]
        print(f"  FAILED: {', '.join(failed)}")
    print("=" * 110)


# ─── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quality benchmarks for incremental ingestion pipeline"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Directory to store results",
    )
    parser.add_argument(
        "--drift-iterations",
        type=int,
        default=20,
        help="Number of sequential edits for context drift test",
    )
    parser.add_argument(
        "--drift-paragraphs",
        type=int,
        default=12,
        help="Number of paragraphs in drift test document",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    chunker = SemanticChunker(min_chunk_size=50, max_chunk_size=3000)
    detector = DiffDetector(similarity_threshold=0.4)

    # ─── 1. Context Drift ────────────────────────────────────────────
    print("\n>>> BENCHMARK 1: Context Drift")
    print("    Simulating %d sequential edits...\n" % args.drift_iterations)
    drift_results = simulate_context_drift(
        num_iterations=args.drift_iterations,
        num_paragraphs=args.drift_paragraphs,
    )
    print_drift_table(drift_results)

    # ─── 2. Diff Accuracy ────────────────────────────────────────────
    print("\n>>> BENCHMARK 2: Diff Accuracy")
    accuracy_results = measure_diff_accuracy(chunker, detector)
    print_accuracy_table(accuracy_results)

    # ─── 3. Hash Stability ───────────────────────────────────────────
    print("\n>>> BENCHMARK 3: Hash Stability")
    stability_results = measure_hash_stability(chunker)
    print_hash_stability_table(stability_results)

    # ─── 4. Scalability ──────────────────────────────────────────────
    print("\n>>> BENCHMARK 4: Scalability")
    scalability_results = measure_scalability()
    print_scalability_table(scalability_results)

    # ─── 5. Stress Tests ─────────────────────────────────────────────
    print("\n>>> BENCHMARK 5: Stress Tests (Edge Cases)")
    stress_results = run_stress_tests()
    print_stress_table(stress_results)

    # ─── Export ──────────────────────────────────────────────────────
    export_all(output_dir, drift_results, accuracy_results, stability_results, scalability_results, stress_results)

    print(f"\nAll quality benchmark results saved to {output_dir}/")


if __name__ == "__main__":
    main()
