#!/usr/bin/env python3
"""Benchmark script for incremental vs full document ingestion.

This script demonstrates and measures the efficiency of the incremental
update pipeline compared to full re-ingestion, as described in the thesis
§Methodologie: Experimenten en evaluatie.

⚠️  IMPORTANT — SIMULATED EMBEDDING TIMES ⚠️
    The embedding times reported by this script are ESTIMATES based on a
    configurable `--time-per-chunk-ms` value (default: 50 ms/chunk).
    They are NOT real wall-clock measurements.
    For real embedding benchmarks, run:
        python -m scripts.benchmark_compute

Experiments performed:
1. Initial ingestion of test documents
2. Small modification (10% change) -> measure incremental vs full
3. Medium modification (50% change) -> measure incremental vs full
4. Large modification (90% change) -> measure incremental vs full
5. No modification -> measure skip detection

What IS measured in real time:
- Chunking time (SemanticChunker)
- Diff detection time (DiffDetector / hash comparison)
- ChunkHashStore read/write performance

What is ESTIMATED (not measured):
- Embedding time (uses time_per_chunk_ms * num_chunks)
- Speedup factor (derived from estimated times)

Output:
- Per-experiment timing and chunk statistics
- Summary table with efficiency ratios
- CSV export for further analysis

Usage:
    python -m scripts.benchmark_incremental [--output-dir ./results]
    python -m scripts.benchmark_incremental --time-per-chunk-ms 100

Note: This script can run standalone without the full PrivateGPT server.
It uses the incremental pipeline components directly.
"""

import argparse
import csv
import json
import logging
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Add the project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from private_gpt.components.ingest.incremental.chunk_hash_store import ChunkHashStore
from private_gpt.components.ingest.incremental.chunk_hasher import (
    HashedChunk,
    SemanticChunker,
)
from private_gpt.components.ingest.incremental.diff_detector import (
    ChangeType,
    DiffDetector,
    patience_diff,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Test Document Generation ────────────────────────────────────────────

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


def generate_document(num_paragraphs: int = 10, seed: int = 42) -> str:
    """Generate a test document with the given number of paragraphs."""
    rng = random.Random(seed)
    paragraphs = [rng.choice(SAMPLE_PARAGRAPHS) for _ in range(num_paragraphs)]
    return "\n\n".join(paragraphs)


def modify_document(
    text: str, change_ratio: float, seed: int = 123
) -> str:
    """Modify a percentage of paragraphs in the document.

    Args:
        text: Original document text.
        change_ratio: Fraction of paragraphs to modify (0.0–1.0).
        seed: Random seed for reproducibility.

    Returns:
        Modified document text.
    """
    rng = random.Random(seed)
    paragraphs = text.split("\n\n")
    num_to_change = max(1, int(len(paragraphs) * change_ratio))
    indices_to_change = rng.sample(range(len(paragraphs)), min(num_to_change, len(paragraphs)))

    APPEND_TEXT = (
        " This is an appended sentence that modifies the contents of this "
        "paragraph with additional information."
    )

    for idx in indices_to_change:
        original = paragraphs[idx]
        modification_type = rng.choice(["append", "replace", "insert"])
        new_paragraph = original

        if modification_type == "append":
            new_paragraph = original + APPEND_TEXT
        elif modification_type == "replace":
            sentences = original.split(". ")
            if len(sentences) > 1:
                replace_idx = rng.randint(0, len(sentences) - 1)
                sentences[replace_idx] = (
                    "This sentence has been entirely replaced with new "
                    "content that differs from the original"
                )
                new_paragraph = ". ".join(sentences)
        elif modification_type == "insert":
            words = original.split()
            if len(words) > 5:
                insert_pos = rng.randint(2, len(words) - 2)
                words.insert(
                    insert_pos,
                    "[NEWLY INSERTED TEXT with extra words for the experiment]",
                )
                new_paragraph = " ".join(words)

        # Fallback: ensure the paragraph genuinely changes even when the
        # chosen modification type would have been a no-op (e.g. "replace"
        # on a single-sentence paragraph or "insert" on a very short one).
        # Without this, a change_ratio of 1.0 can still leave several
        # paragraphs untouched, which inflates UNCHANGED counts in the
        # benchmark and makes the "Complete replacement" scenario report
        # an artificially high reuse rate.
        if new_paragraph == original:
            new_paragraph = original + APPEND_TEXT

        paragraphs[idx] = new_paragraph

    return "\n\n".join(paragraphs)


# ─── Benchmark Functions ─────────────────────────────────────────────────


def benchmark_chunking(
    chunker: SemanticChunker, text: str, label: str
) -> tuple[list[HashedChunk], float]:
    """Benchmark the chunking operation."""
    start = time.perf_counter()
    chunks = chunker.chunk_text(text, metadata={"label": label})
    elapsed = time.perf_counter() - start
    return chunks, elapsed


def benchmark_diff(
    detector: DiffDetector,
    old_chunks: list[HashedChunk],
    new_chunks: list[HashedChunk],
) -> tuple[dict, float]:
    """Benchmark the diff detection."""
    start = time.perf_counter()
    changes = detector.detect_changes(old_chunks, new_chunks)
    elapsed = time.perf_counter() - start

    stats = {
        "added": sum(1 for c in changes if c.change_type == ChangeType.ADDED),
        "modified": sum(1 for c in changes if c.change_type == ChangeType.MODIFIED),
        "deleted": sum(1 for c in changes if c.change_type == ChangeType.DELETED),
        "unchanged": sum(1 for c in changes if c.change_type == ChangeType.UNCHANGED),
    }
    return stats, elapsed


def simulate_full_reingest_time(num_chunks: int, time_per_chunk_ms: float = 50.0) -> float:
    """Simulate the time a full re-ingest would take.

    Based on the thesis measurements: embedding each chunk takes  50ms
    on typical consumer hardware.
    """
    return num_chunks * time_per_chunk_ms / 1000.0


def run_experiment(
    name: str,
    original_text: str,
    modified_text: str,
    chunker: SemanticChunker,
    detector: DiffDetector,
    time_per_chunk_ms: float = 50.0,
) -> dict:
    """Run a single benchmark experiment.

    Args:
        name: Experiment name.
        original_text: Original document text.
        modified_text: Modified document text.
        chunker: SemanticChunker instance.
        detector: DiffDetector instance.
        time_per_chunk_ms: Simulated time per embedding computation.

    Returns:
        Dictionary with experiment results.
    """
    logger.info("=" * 60)
    logger.info("Running experiment: %s", name)
    logger.info("=" * 60)

    # Chunk original
    old_chunks, chunk_time_old = benchmark_chunking(chunker, original_text, "old")
    logger.info("  Original: %d chunks (%.3fs)", len(old_chunks), chunk_time_old)

    # Chunk modified
    new_chunks, chunk_time_new = benchmark_chunking(chunker, modified_text, "new")
    logger.info("  Modified: %d chunks (%.3fs)", len(new_chunks), chunk_time_new)

    # Detect differences
    diff_stats, diff_time = benchmark_diff(detector, old_chunks, new_chunks)
    logger.info("  Diff detection: %.3fs", diff_time)
    logger.info(
        "  Changes: %d unchanged, %d modified, %d added, %d deleted",
        diff_stats["unchanged"],
        diff_stats["modified"],
        diff_stats["added"],
        diff_stats["deleted"],
    )

    # Calculate incremental vs full cost
    chunks_to_embed = diff_stats["modified"] + diff_stats["added"]
    incremental_embed_time = simulate_full_reingest_time(
        chunks_to_embed, time_per_chunk_ms
    )
    full_reingest_time = simulate_full_reingest_time(
        len(new_chunks), time_per_chunk_ms
    )

    total_incremental_time = chunk_time_new + diff_time + incremental_embed_time
    efficiency = (
        1 - (chunks_to_embed / len(new_chunks)) if len(new_chunks) > 0 else 0
    )

    speedup = full_reingest_time / total_incremental_time if total_incremental_time > 0 else float("inf")

    logger.info("  Embeddings: %d computed, %d skipped", chunks_to_embed, diff_stats["unchanged"])
    logger.info("  Efficiency: %.1f%% reuse", efficiency * 100)
    logger.info("  Incremental time: %.3fs (simulated)", total_incremental_time)
    logger.info("  Full re-ingest time: %.3fs (simulated)", full_reingest_time)
    logger.info("  Speedup: %.1fx", speedup)

    return {
        "experiment": name,
        "chunks_old": len(old_chunks),
        "chunks_new": len(new_chunks),
        "unchanged": diff_stats["unchanged"],
        "modified": diff_stats["modified"],
        "added": diff_stats["added"],
        "deleted": diff_stats["deleted"],
        "embeddings_computed": chunks_to_embed,
        "embeddings_skipped": diff_stats["unchanged"],
        "efficiency_ratio": efficiency,
        "time_chunking_s": chunk_time_new,
        "time_diffing_s": diff_time,
        # NOTE: the two fields below are ESTIMATED, not measured
        "time_incremental_estimated_s": total_incremental_time,
        "time_full_reingest_estimated_s": full_reingest_time,
        "speedup_factor": speedup,
    }


def benchmark_patience_diff(original_text: str, modified_text: str) -> dict:
    """Benchmark the Patience diff algorithm on raw text."""
    old_lines = original_text.splitlines()
    new_lines = modified_text.splitlines()

    start = time.perf_counter()
    result = patience_diff(old_lines, new_lines)
    elapsed = time.perf_counter() - start

    additions = sum(1 for tag, _ in result if tag == "+")
    deletions = sum(1 for tag, _ in result if tag == "-")
    context = sum(1 for tag, _ in result if tag == " ")

    return {
        "time_s": elapsed,
        "additions": additions,
        "deletions": deletions,
        "context_lines": context,
        "total_lines": len(result),
    }


def benchmark_hash_store(persist_dir: str, num_docs: int = 100) -> dict:
    """Benchmark the ChunkHashStore with many documents."""
    store = ChunkHashStore(persist_dir=persist_dir)

    from private_gpt.components.ingest.incremental.chunk_hash_store import (
        DocumentRecord,
        StoredChunkInfo,
    )

    # Write documents
    start = time.perf_counter()
    for i in range(num_docs):
        chunks = [
            StoredChunkInfo(
                chunk_index=j,
                content_hash=f"hash_{i}_{j}",
                node_id=f"node_{i}_{j}",
                text_preview=f"Preview text for doc {i} chunk {j}",
            )
            for j in range(10)
        ]
        record = DocumentRecord(
            doc_id=f"doc_{i}",
            file_name=f"document_{i}.txt",
            file_hash=f"filehash_{i}",
            chunks=chunks,
        )
        store.upsert_document(record)
    write_time = time.perf_counter() - start

    # Read documents
    start = time.perf_counter()
    for i in range(num_docs):
        store.get_document(f"doc_{i}")
    read_time = time.perf_counter() - start

    # Lookup by filename
    start = time.perf_counter()
    for i in range(num_docs):
        store.get_document_by_filename(f"document_{i}.txt")
    lookup_time = time.perf_counter() - start

    return {
        "num_docs": num_docs,
        "write_time_s": write_time,
        "read_time_s": read_time,
        "lookup_time_s": lookup_time,
        "time_per_write_ms": (write_time / num_docs) * 1000,
        "time_per_read_ms": (read_time / num_docs) * 1000,
    }


# ─── Main ────────────────────────────────────────────────────────────────


def print_summary_table(results: list[dict]) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS SUMMARY")
    print("** Embed times and Speedup are ESTIMATED (50 ms/chunk default). **")
    print("   Run benchmark_compute.py for real wall-clock measurements.")
    print("=" * 90)
    print(
        f"{'Experiment':<25} {'Old':>5} {'New':>5} {'Unchg':>5} {'Mod':>5} "
        f"{'Add':>5} {'Del':>5} {'Eff%':>6} {'Speedup*':>9}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r['experiment']:<25} "
            f"{r['chunks_old']:>5} "
            f"{r['chunks_new']:>5} "
            f"{r['unchanged']:>5} "
            f"{r['modified']:>5} "
            f"{r['added']:>5} "
            f"{r['deleted']:>5} "
            f"{r['efficiency_ratio']*100:>5.1f}% "
            f"{r['speedup_factor']:>8.1f}x"
        )
    print("=" * 90)
    print("* Speedup is estimated from simulated embed times, not measured.")


def export_csv(results: list[dict], output_path: Path) -> None:
    """Export results as CSV."""
    if not results:
        return
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    logger.info("Results exported to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark incremental vs full document ingestion"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Directory to store benchmark results",
    )
    parser.add_argument(
        "--num-paragraphs",
        type=int,
        default=15,
        help="Number of paragraphs in test documents",
    )
    parser.add_argument(
        "--time-per-chunk-ms",
        type=float,
        default=50.0,
        help="Simulated embedding time per chunk in milliseconds",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunker = SemanticChunker(min_chunk_size=100, max_chunk_size=3000)
    detector = DiffDetector(similarity_threshold=0.4)

    # ─── WARM-UP STEP ───────────────────────────────────────────────
    logger.info("Initializing and warming up embedding model...")
    _ = chunker.chunk_text("Warming up the embedding model")

    # Generate the original document
    original = generate_document(num_paragraphs=args.num_paragraphs, seed=42)
    logger.info("Generated document with %d characters", len(original))

    results = []

    # ─── Experiment 1: No change (should skip entirely) ──────────────
    result = run_experiment(
        name="No change (0%)",
        original_text=original,
        modified_text=original,  # Same text
        chunker=chunker,
        detector=detector,
        time_per_chunk_ms=args.time_per_chunk_ms,
    )
    results.append(result)

    # ─── Experiment 2: Small modification (10%) ──────────────────────
    modified_10 = modify_document(original, change_ratio=0.10, seed=100)
    result = run_experiment(
        name="Small change (10%)",
        original_text=original,
        modified_text=modified_10,
        chunker=chunker,
        detector=detector,
        time_per_chunk_ms=args.time_per_chunk_ms,
    )
    results.append(result)

    # ─── Experiment 3: Medium modification (30%) ─────────────────────
    modified_30 = modify_document(original, change_ratio=0.30, seed=200)
    result = run_experiment(
        name="Medium change (30%)",
        original_text=original,
        modified_text=modified_30,
        chunker=chunker,
        detector=detector,
        time_per_chunk_ms=args.time_per_chunk_ms,
    )
    results.append(result)

    # ─── Experiment 4: Large modification (50%) ──────────────────────
    modified_50 = modify_document(original, change_ratio=0.50, seed=300)
    result = run_experiment(
        name="Large change (50%)",
        original_text=original,
        modified_text=modified_50,
        chunker=chunker,
        detector=detector,
        time_per_chunk_ms=args.time_per_chunk_ms,
    )
    results.append(result)

    # ─── Experiment 5: Very large modification (90%) ─────────────────
    modified_90 = modify_document(original, change_ratio=0.90, seed=400)
    result = run_experiment(
        name="Very large change (90%)",
        original_text=original,
        modified_text=modified_90,
        chunker=chunker,
        detector=detector,
        time_per_chunk_ms=args.time_per_chunk_ms,
    )
    results.append(result)

    # ─── Experiment 6: Complete replacement (100%) ───────────────────
    # Modify every paragraph so no chunk hash can match the original.
    # Using generate_document(seed=999) would reuse the same paragraph
    # pool and accidentally share  40% of hashes — that is not "complete
    # replacement."
    completely_new = modify_document(original, change_ratio=1.0, seed=999)
    result = run_experiment(
        name="Complete replacement",
        original_text=original,
        modified_text=completely_new,
        chunker=chunker,
        detector=detector,
        time_per_chunk_ms=args.time_per_chunk_ms,
    )
    results.append(result)

    # ─── Print summary ───────────────────────────────────────────────
    print_summary_table(results)

    # ─── Patience diff benchmark ─────────────────────────────────────
    print("\n--- Patience Diff Benchmark ---")
    for name, mod_text in [
        ("10% change", modified_10),
        ("50% change", modified_50),
        ("90% change", modified_90),
    ]:
        pd_result = benchmark_patience_diff(original, mod_text)
        print(
            f"  {name}: {pd_result['time_s']:.4f}s "
            f"(+{pd_result['additions']} -{pd_result['deletions']} "
            f"={pd_result['context_lines']} context)"
        )

    # ─── Hash store benchmark ────────────────────────────────────────
    print("\n--- Hash Store Benchmark ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        hs_result = benchmark_hash_store(tmpdir, num_docs=100)
        print(
            f"  {hs_result['num_docs']} docs: "
            f"write={hs_result['time_per_write_ms']:.2f}ms/doc, "
            f"read={hs_result['time_per_read_ms']:.2f}ms/doc"
        )

    # ─── Export results ──────────────────────────────────────────────
    export_csv(results, output_dir / "benchmark_results.csv")

    # Also export as JSON for programmatic access
    with open(output_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Full results exported to %s", output_dir)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
