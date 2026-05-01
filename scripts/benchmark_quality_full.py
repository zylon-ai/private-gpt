#!/usr/bin/env python3
"""Quality benchmarks for the full re-ingest baseline.

This is the baseline counterpart to ``benchmark_quality.py``. Where the
incremental variant measures the *stability* of chunk embeddings across
edits, this script measures the *inefficiency* of the default full
re-ingest pipeline:

1. Inefficiency Ratio – Fraction of chunks that are byte-identical
   across document versions but still re-embedded because the pipeline
   has no diff step.  With a fixed-size chunker this is typically close
   to 0.0 (avalanche effect); with a semantic chunker it is higher, but
   the pipeline still re-embeds everything.

2. Avalanche Ratio – Fraction of chunks whose content hash changed
   between old and new versions, even though only a small part of the
   document was actually edited.  Quantifies the "domino effect" of
   fixed-size chunking and motivates semantic chunking as a separate
   (but complementary) improvement to incremental updates.

3. Hash Stability – Same chunker-level stability test as in
   benchmark_quality.py.  Chunker behaviour is independent of the
   ingest pipeline.

4. Scalability – Chunking throughput at different document sizes for
   both chunkers; shows the constant-factor difference.

Output files are written with the ``_full`` suffix so they can be
plotted side-by-side with the incremental results:

  quality_full_inefficiency.{csv,json}
  quality_full_avalanche.{csv,json}
  quality_full_stability.{csv,json}
  quality_full_scalability.{csv,json}

Usage:
    python -m scripts.benchmark_quality_full [--output-dir ./benchmark_results]

References (from thesis):
- Problem statement: avalanche effect of fixed-size chunking
- Methodology: baseline comparison with full re-ingest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_quality import (
    SAMPLE_PARAGRAPHS,
    delete_paragraph,
    generate_document,
    insert_paragraph,
    modify_single_paragraph,
)
from private_gpt.components.ingest.incremental.chunk_hasher import SemanticChunker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Fixed-size chunker (PrivateGPT baseline) ───────────────────────────


@dataclass
class FixedSizeChunk:
    """A chunk produced by the fixed-size baseline chunker."""

    chunk_index: int
    text: str
    content_hash: str


class FixedSizeChunker:
    """Character-window chunker that emulates the default PrivateGPT
    ``SentenceSplitter(chunk_size=1024, chunk_overlap=20)`` in character
    space. Intentionally simple — no sentence awareness, so the
    avalanche effect is visible at any edit position."""

    def __init__(self, chunk_size: int = 1024, overlap: int = 20) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(
        self, text: str, metadata: dict | None = None
    ) -> list[FixedSizeChunk]:
        del metadata  # unused — included for interface parity
        step = max(1, self.chunk_size - self.overlap)
        chunks: list[FixedSizeChunk] = []
        idx = 0
        pos = 0
        while pos < len(text):
            body = text[pos : pos + self.chunk_size]
            if body.strip():
                normalised = re.sub(r"\s+", " ", body).strip()
                h = hashlib.sha256(normalised.encode("utf-8")).hexdigest()
                chunks.append(
                    FixedSizeChunk(chunk_index=idx, text=body, content_hash=h)
                )
                idx += 1
            pos += step
        return chunks


# ─── Benchmark 1: Full re-ingest inefficiency ───────────────────────────


@dataclass
class InefficiencyMeasurement:
    """For a single sequential edit, count how many chunks were byte-
    identical with the previous version but were still re-embedded
    because the full pipeline has no diff step."""

    iteration: int
    edit_type: str
    chunker: str
    total_chunks_new: int
    chunks_hash_preserved: int
    chunks_reembedded_unnecessarily: int
    inefficiency_ratio: float


def measure_full_reingest_inefficiency(
    num_iterations: int,
    num_paragraphs: int,
    seed: int = 42,
) -> list[InefficiencyMeasurement]:
    """Sequential edits; for each version, count chunks whose hash matches
    the previous version — those are the chunks the full pipeline would
    re-embed uselessly.

    Both chunkers receive the SAME sequence of text versions so that the
    edit labels on the x-axis match both lines correctly.
    """
    rng = random.Random(seed)
    results: list[InefficiencyMeasurement] = []

    # Build the full edit plan once: list of (iteration, edit_type, text_after_edit)
    # Both chunkers replay the same text snapshots — only how they chunk differs.
    edit_plan: list[tuple[int, str, str]] = []
    text = generate_document(num_paragraphs=num_paragraphs, seed=seed)
    for iteration in range(1, num_iterations + 1):
        edit_type = rng.choice(["modify", "insert", "delete"])
        paras = text.split("\n\n")
        if edit_type == "modify":
            idx = rng.randint(0, len(paras) - 1)
            text = modify_single_paragraph(text, idx, seed=iteration * 100)
        elif edit_type == "insert":
            idx = rng.randint(0, len(paras))
            text = insert_paragraph(text, idx, seed=iteration * 100)
        elif edit_type == "delete" and len(paras) > 3:
            idx = rng.randint(0, len(paras) - 1)
            text = delete_paragraph(text, idx)
        else:
            edit_type = "modify"
            text = modify_single_paragraph(text, 0, seed=iteration * 100)
        edit_plan.append((iteration, edit_type, text))

    chunker_pairs = [
        ("fixed-size (1024/20)", FixedSizeChunker(1024, 20)),
        ("semantic", SemanticChunker(min_chunk_size=50, max_chunk_size=3000)),
    ]

    base_text = generate_document(num_paragraphs=num_paragraphs, seed=seed)
    for chunker_label, chunker in chunker_pairs:
        prev_hashes = {c.content_hash for c in chunker.chunk_text(base_text)}

        for iteration, edit_type, edited_text in edit_plan:
            new_chunks = chunker.chunk_text(edited_text)
            new_hashes = [c.content_hash for c in new_chunks]
            preserved = sum(1 for h in new_hashes if h in prev_hashes)
            total = len(new_chunks)
            ineff = preserved / total if total > 0 else 0.0

            results.append(
                InefficiencyMeasurement(
                    iteration=iteration,
                    edit_type=edit_type,
                    chunker=chunker_label,
                    total_chunks_new=total,
                    chunks_hash_preserved=preserved,
                    chunks_reembedded_unnecessarily=preserved,
                    inefficiency_ratio=ineff,
                )
            )
            logger.info(
                "  [%s] Iter %2d [%s]: %d chunks, "
                "%d preserved hashes, inefficiency=%.1f%%",
                chunker_label, iteration, edit_type, total, preserved,
                ineff * 100,
            )
            prev_hashes = set(new_hashes)

    return results


# ─── Benchmark 2: Avalanche ratio ───────────────────────────────────────


@dataclass
class AvalancheMeasurement:
    """Fraction of chunks whose hash changed, given that only one
    paragraph was actually edited. For a stable chunker this is  1/N;
    for a fixed-size chunker near the edit site it grows toward 1.0."""

    scenario: str
    chunker: str
    num_paragraphs: int
    num_chunks_old: int
    num_chunks_new: int
    chunks_hash_changed: int
    avalanche_ratio: float
    expected_change_fraction: float


def measure_avalanche_effect(
    num_paragraphs: int = 15,
    seed: int = 42,
) -> list[AvalancheMeasurement]:
    """Compare avalanche behaviour of fixed-size vs semantic chunkers on
    the same single-paragraph edit."""
    results: list[AvalancheMeasurement] = []
    base_text = generate_document(num_paragraphs=num_paragraphs, seed=seed)

    chunker_pairs = [
        ("fixed-size (1024/20)", FixedSizeChunker(1024, 20)),
        ("semantic", SemanticChunker(min_chunk_size=50, max_chunk_size=3000)),
    ]

    scenarios = [
        ("modify first paragraph", 0, "modify"),
        ("modify middle paragraph", num_paragraphs // 2, "modify"),
        ("modify last paragraph", num_paragraphs - 1, "modify"),
        ("insert at start", 0, "insert"),
        ("insert in middle", num_paragraphs // 2, "insert"),
        ("delete first paragraph", 0, "delete"),
        ("delete middle paragraph", num_paragraphs // 2, "delete"),
    ]

    for scenario_name, idx, edit_type in scenarios:
        if edit_type == "modify":
            modified = modify_single_paragraph(base_text, idx, seed=idx)
        elif edit_type == "insert":
            modified = insert_paragraph(base_text, idx, seed=idx)
        else:
            modified = delete_paragraph(base_text, idx)

        for chunker_label, chunker in chunker_pairs:
            old_chunks = chunker.chunk_text(base_text)
            new_chunks = chunker.chunk_text(modified)
            old_hashes = {c.content_hash for c in old_chunks}
            changed = sum(
                1 for c in new_chunks if c.content_hash not in old_hashes
            )
            ratio = changed / len(new_chunks) if new_chunks else 0.0
            expected = 1.0 / num_paragraphs

            results.append(
                AvalancheMeasurement(
                    scenario=scenario_name,
                    chunker=chunker_label,
                    num_paragraphs=num_paragraphs,
                    num_chunks_old=len(old_chunks),
                    num_chunks_new=len(new_chunks),
                    chunks_hash_changed=changed,
                    avalanche_ratio=ratio,
                    expected_change_fraction=expected,
                )
            )
            logger.info(
                "  [%s] %s: %d/%d chunks changed (ratio=%.2f, "
                "expected=%.2f, amplification=%.1fx)",
                chunker_label, scenario_name, changed, len(new_chunks),
                ratio, expected, ratio / expected if expected > 0 else 0,
            )

    return results


# ─── Benchmark 3: Hash stability (chunker property) ─────────────────────


@dataclass
class HashStabilityResult:
    """Chunker-level stability: identical text → identical hash."""

    test_name: str
    chunker: str
    is_stable: bool


def measure_hash_stability() -> list[HashStabilityResult]:
    """Run the same stability tests on both chunkers so the difference
    is visible in the side-by-side output."""
    results: list[HashStabilityResult] = []

    chunker_pairs = [
        ("fixed-size (1024/20)", FixedSizeChunker(1024, 20)),
        ("semantic", SemanticChunker(min_chunk_size=50, max_chunk_size=3000)),
    ]

    target = SAMPLE_PARAGRAPHS[5]
    ctx_a = "\n\n".join(SAMPLE_PARAGRAPHS[0:5] + [target] + SAMPLE_PARAGRAPHS[6:9])
    ctx_b = "\n\n".join(SAMPLE_PARAGRAPHS[9:12] + [target] + SAMPLE_PARAGRAPHS[12:15])

    for chunker_label, chunker in chunker_pairs:
        chunks_a = chunker.chunk_text(ctx_a)
        chunks_b = chunker.chunk_text(ctx_b)
        hashes_a = {c.content_hash for c in chunks_a}
        hashes_b = {c.content_hash for c in chunks_b}

        target_hash_found_in_both = any(
            target[:50] in c.text for c in chunks_a
        ) and any(target[:50] in c.text for c in chunks_b)
        shared = hashes_a & hashes_b
        is_stable = target_hash_found_in_both and len(shared) > 0

        results.append(
            HashStabilityResult(
                test_name="Same paragraph, different surrounding context",
                chunker=chunker_label,
                is_stable=is_stable,
            )
        )

    return results


# ─── Benchmark 4: Scalability ───────────────────────────────────────────


@dataclass
class ScalabilityResult:
    """Chunking throughput at different document sizes."""

    num_paragraphs: int
    chunker: str
    num_chunks: int
    time_chunking_s: float
    chunks_per_second: float


def measure_scalability(sizes: list[int] | None = None) -> list[ScalabilityResult]:
    if sizes is None:
        sizes = [5, 10, 20, 50, 100, 200]

    chunker_pairs = [
        ("fixed-size (1024/20)", FixedSizeChunker(1024, 20)),
        ("semantic", SemanticChunker(min_chunk_size=50, max_chunk_size=3000)),
    ]

    results: list[ScalabilityResult] = []
    for n in sizes:
        text = generate_document(num_paragraphs=n, seed=42)
        for chunker_label, chunker in chunker_pairs:
            t0 = time.perf_counter()
            chunks = chunker.chunk_text(text)
            t = time.perf_counter() - t0
            cps = len(chunks) / t if t > 0 else 0.0
            results.append(
                ScalabilityResult(
                    num_paragraphs=n,
                    chunker=chunker_label,
                    num_chunks=len(chunks),
                    time_chunking_s=t,
                    chunks_per_second=cps,
                )
            )
            logger.info(
                "  [%s] %3d paras: %d chunks in %.4fs (%.0f chunks/s)",
                chunker_label, n, len(chunks), t, cps,
            )
    return results


# ─── Output / Export ────────────────────────────────────────────────────


def print_inefficiency_table(results: list[InefficiencyMeasurement]) -> None:
    print("\n" + "=" * 100)
    print("FULL RE-INGEST INEFFICIENCY (lower = fewer unnecessary re-embeds)")
    print("=" * 100)
    print(
        f"{'Iter':>4} {'Edit':>8} {'Chunker':<22} "
        f"{'New':>5} {'Preserved':>10} {'Wasted':>7} {'Inefficiency':>12}"
    )
    print("-" * 100)
    for r in results:
        print(
            f"{r.iteration:>4} {r.edit_type:>8} {r.chunker:<22} "
            f"{r.total_chunks_new:>5} {r.chunks_hash_preserved:>10} "
            f"{r.chunks_reembedded_unnecessarily:>7} "
            f"{r.inefficiency_ratio:>11.1%}"
        )
    print("=" * 100)
    print(
        "Note: A chunk whose hash matches the previous version is "
        "re-embedded anyway in the full pipeline — that is the cost "
        "the incremental pipeline eliminates."
    )


def print_avalanche_table(results: list[AvalancheMeasurement]) -> None:
    print("\n" + "=" * 100)
    print("AVALANCHE EFFECT (ratio of chunks changed per single-paragraph edit)")
    print("=" * 100)
    print(
        f"{'Scenario':<32} {'Chunker':<22} {'Old':>4} {'New':>4} "
        f"{'Chg':>4} {'Ratio':>7} {'Expected':>9} {'Amp':>5}"
    )
    print("-" * 100)
    for r in results:
        amp = (
            r.avalanche_ratio / r.expected_change_fraction
            if r.expected_change_fraction > 0 else 0.0
        )
        print(
            f"{r.scenario:<32} {r.chunker:<22} "
            f"{r.num_chunks_old:>4} {r.num_chunks_new:>4} "
            f"{r.chunks_hash_changed:>4} {r.avalanche_ratio:>6.1%} "
            f"{r.expected_change_fraction:>8.1%} {amp:>4.1f}x"
        )
    print("=" * 100)
    print("Amp = avalanche ratio / expected change fraction; 1.0 = no avalanche.")


def print_stability_table(results: list[HashStabilityResult]) -> None:
    print("\n" + "=" * 100)
    print("HASH STABILITY (chunker-level; independent of ingest pipeline)")
    print("=" * 100)
    for r in results:
        status = "STABLE" if r.is_stable else "UNSTABLE"
        print(f"  {r.chunker:<22} {r.test_name}: {status}")
    print("=" * 100)


def print_scalability_table(results: list[ScalabilityResult]) -> None:
    print("\n" + "=" * 100)
    print("CHUNKING SCALABILITY (throughput vs document size)")
    print("=" * 100)
    print(f"{'Paras':>6} {'Chunker':<22} {'Chunks':>7} "
          f"{'Time(s)':>9} {'Chunks/s':>10}")
    print("-" * 100)
    for r in results:
        print(
            f"{r.num_paragraphs:>6} {r.chunker:<22} {r.num_chunks:>7} "
            f"{r.time_chunking_s:>9.4f} {r.chunks_per_second:>10.0f}"
        )
    print("=" * 100)


def export_all(
    output_dir: Path,
    inefficiency: list[InefficiencyMeasurement],
    avalanche: list[AvalancheMeasurement],
    stability: list[HashStabilityResult],
    scalability: list[ScalabilityResult],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    def _write(name: str, rows: list[dict]) -> None:
        with open(output_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        if rows:
            with open(
                output_dir / f"{name}.csv", "w", newline="", encoding="utf-8"
            ) as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)

    _write(
        "quality_full_inefficiency",
        [
            {
                "iteration": r.iteration,
                "edit_type": r.edit_type,
                "chunker": r.chunker,
                "total_chunks_new": r.total_chunks_new,
                "chunks_hash_preserved": r.chunks_hash_preserved,
                "chunks_reembedded_unnecessarily": r.chunks_reembedded_unnecessarily,
                "inefficiency_ratio": r.inefficiency_ratio,
            }
            for r in inefficiency
        ],
    )
    _write(
        "quality_full_avalanche",
        [
            {
                "scenario": r.scenario,
                "chunker": r.chunker,
                "num_paragraphs": r.num_paragraphs,
                "num_chunks_old": r.num_chunks_old,
                "num_chunks_new": r.num_chunks_new,
                "chunks_hash_changed": r.chunks_hash_changed,
                "avalanche_ratio": r.avalanche_ratio,
                "expected_change_fraction": r.expected_change_fraction,
            }
            for r in avalanche
        ],
    )
    _write(
        "quality_full_stability",
        [
            {
                "test_name": r.test_name,
                "chunker": r.chunker,
                "is_stable": r.is_stable,
            }
            for r in stability
        ],
    )
    _write(
        "quality_full_scalability",
        [
            {
                "num_paragraphs": r.num_paragraphs,
                "chunker": r.chunker,
                "num_chunks": r.num_chunks,
                "time_chunking_s": r.time_chunking_s,
                "chunks_per_second": r.chunks_per_second,
            }
            for r in scalability
        ],
    )
    logger.info("Exported full re-ingest quality results to %s", output_dir)


# ─── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quality baseline benchmarks for the full re-ingest pipeline",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Directory to store results",
    )
    parser.add_argument(
        "--iterations", type=int, default=10,
        help="Number of sequential edits in the inefficiency benchmark",
    )
    parser.add_argument(
        "--paragraphs", type=int, default=12,
        help="Document size (paragraphs) for the inefficiency benchmark",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print("\n>>> BENCHMARK 1: Full re-ingest inefficiency")
    ineff = measure_full_reingest_inefficiency(
        num_iterations=args.iterations, num_paragraphs=args.paragraphs,
    )
    print_inefficiency_table(ineff)

    print("\n>>> BENCHMARK 2: Avalanche effect")
    aval = measure_avalanche_effect()
    print_avalanche_table(aval)

    print("\n>>> BENCHMARK 3: Hash stability (chunker property)")
    stab = measure_hash_stability()
    print_stability_table(stab)

    print("\n>>> BENCHMARK 4: Chunking scalability")
    scale = measure_scalability()
    print_scalability_table(scale)

    export_all(output_dir, ineff, aval, stab, scale)
    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
