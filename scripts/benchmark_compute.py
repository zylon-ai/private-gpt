#!/usr/bin/env python3
"""Computational performance benchmark for incremental vs full ingestion.

Measures REAL resource usage (not simulated):
  - Wall-clock time for chunking, diffing, and embedding
  - Peak memory usage (RSS)
  - CPU time (user + system)
  - Per-chunk embedding throughput

Designed to run on any machine (laptop, VM, server) and produce
reproducible results.  For best results, run on an isolated VM with
no other workloads.

Usage:
    poetry run python -m scripts.benchmark_compute [--runs 3]

Output:
    benchmark_results/compute_benchmark.json
    benchmark_results/compute_benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Platform info ─────────────────────────────────────────────────────

def get_system_info() -> dict:
    """Collect system information for reproducibility."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["total_memory_gb"] = round(mem.total / (1024**3), 2)
    except ImportError:
        info["total_memory_gb"] = "unknown (install psutil)"
    return info


def get_memory_mb() -> float:
    """Get current process RSS in MB."""
    try:
        import psutil  # type: ignore
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback for Windows without psutil
        try:
            import ctypes
            import ctypes.wintypes
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.wintypes.DWORD),
                    ("PageFaultCount", ctypes.wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(pmc), pmc.cb
            )
            return pmc.WorkingSetSize / (1024 * 1024)
        except Exception:
            return 0.0


# ─── Test corpus ───────────────────────────────────────────────────────

def load_test_documents(test_dir: Path) -> dict[str, str]:
    """Load test documents from the test_documents directory."""
    docs = {}
    for name in ["base_document.txt", "modified_10pct.txt",
                  "modified_50pct.txt", "modified_90pct.txt"]:
        path = test_dir / name
        if path.exists():
            docs[name] = path.read_text(encoding="utf-8")
        else:
            logger.warning("Test document not found: %s", path)
    return docs


# ─── Benchmark runner ──────────────────────────────────────────────────

def benchmark_single_run(
    base_text: str,
    modified_text: str,
    label: str,
    embedding_model_name: str = "nomic-ai/nomic-embed-text-v1.5",
) -> dict:
    """Run a single benchmark: chunk, diff, embed, measure resources."""
    from private_gpt.components.ingest.incremental.chunk_hasher import (
        SemanticChunker,
    )
    from private_gpt.components.ingest.incremental.diff_detector import (
        ChangeType,
        DiffDetector,
    )

    chunker = SemanticChunker(min_chunk_size=100, max_chunk_size=3000)
    detector = DiffDetector(similarity_threshold=0.4)

    mem_before = get_memory_mb()
    cpu_before = time.process_time()
    wall_start = time.perf_counter()

    # Step 1: Chunk base
    t0 = time.perf_counter()
    old_chunks = chunker.chunk_text(base_text, metadata={"version": "old"})
    time_chunk_old = time.perf_counter() - t0

    # Step 2: Chunk modified
    t0 = time.perf_counter()
    new_chunks = chunker.chunk_text(modified_text, metadata={"version": "new"})
    time_chunk_new = time.perf_counter() - t0

    # Step 3: Diff
    t0 = time.perf_counter()
    changes = detector.detect_changes(old_chunks, new_chunks)
    time_diff = time.perf_counter() - t0

    added = [c for c in changes if c.change_type == ChangeType.ADDED]
    modified = [c for c in changes if c.change_type == ChangeType.MODIFIED]
    deleted = [c for c in changes if c.change_type == ChangeType.DELETED]
    unchanged = [c for c in changes if c.change_type == ChangeType.UNCHANGED]

    chunks_to_embed = len(added) + len(modified)

    # Step 4: Actual embedding (real computation, not simulated!)
    embed_texts = []
    for c in added:
        if c.new_chunk:
            embed_texts.append(c.new_chunk.text)
    for c in modified:
        if c.new_chunk:
            embed_texts.append(c.new_chunk.text)

    time_embed_incremental = 0.0
    if embed_texts:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            model = SentenceTransformer(
                embedding_model_name, trust_remote_code=True
            )
            t0 = time.perf_counter()
            model.encode(embed_texts, show_progress_bar=False,
                         normalize_embeddings=True)
            time_embed_incremental = time.perf_counter() - t0
        except ImportError:
            logger.warning("sentence-transformers not available, skipping embed")
            time_embed_incremental = -1.0

    # Step 5: Full re-embed with Basic Chunking (PrivateGPT baseline)
    # Chunking and model loading are done BEFORE starting the timer so that
    # only encode() is measured — matching the incremental path above.
    try:
        from llama_index.core.node_parser import SentenceSplitter  # type: ignore
        from llama_index.core import Document  # type: ignore
        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        nodes = splitter.get_nodes_from_documents([Document(text=modified_text)])
        all_texts = [n.get_content() for n in nodes]
    except ImportError:
        # Fallback crude method if LlamaIndex is unavailable
        all_texts = [str(modified_text)[i:i+1024] for i in range(0, len(modified_text), 1024-20)]

    time_embed_full = 0.0
    if all_texts:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            model = SentenceTransformer(
                embedding_model_name, trust_remote_code=True
            )
            # Timer starts here — only the encode call is measured,
            # identical to how time_embed_incremental is measured above.
            t_baseline_start = time.perf_counter()
            model.encode(all_texts, show_progress_bar=False,
                         normalize_embeddings=True)
            time_embed_full = time.perf_counter() - t_baseline_start
        except ImportError:
            time_embed_full = -1.0

    wall_total = time.perf_counter() - wall_start
    cpu_total = time.process_time() - cpu_before
    mem_after = get_memory_mb()

    speedup = (
        time_embed_full / time_embed_incremental
        if time_embed_incremental > 0 else float("inf")
    )

    return {
        "label": label,
        "chunks_old": len(old_chunks),
        "chunks_new": len(new_chunks),
        "unchanged": len(unchanged),
        "modified": len(modified),
        "added": len(added),
        "deleted": len(deleted),
        "chunks_embedded_incremental": chunks_to_embed,
        "chunks_embedded_full": len(new_chunks),
        "time_chunk_old_s": round(time_chunk_old, 4),
        "time_chunk_new_s": round(time_chunk_new, 4),
        "time_diff_s": round(time_diff, 4),
        "time_embed_incremental_s": round(time_embed_incremental, 4),
        "time_embed_full_s": round(time_embed_full, 4),
        "time_wall_total_s": round(wall_total, 4),
        "time_cpu_total_s": round(cpu_total, 4),
        "memory_before_mb": round(mem_before, 1),
        "memory_after_mb": round(mem_after, 1),
        "memory_delta_mb": round(mem_after - mem_before, 1),
        "embed_speedup": round(speedup, 2),
        "efficiency_pct": round(
            (1 - chunks_to_embed / len(new_chunks)) * 100
            if len(new_chunks) > 0 else 0, 1
        ),
    }


def run_all_benchmarks(
    test_dir: Path,
    num_runs: int = 3,
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
) -> dict:
    """Run benchmarks for all change levels, averaged over num_runs."""
    docs = load_test_documents(test_dir)

    base = docs.get("base_document.txt")
    if not base:
        logger.error("base_document.txt not found in %s", test_dir)
        sys.exit(1)

    scenarios = [
        ("0% change", base, base),
        ("10% change", base, docs.get("modified_10pct.txt", base)),
        ("50% change", base, docs.get("modified_50pct.txt", base)),
        ("90% change", base, docs.get("modified_90pct.txt", base)),
    ]

    # --- WARM-UP STEP ---
    # The first model execution suffers from JIT compilation and memory loading (Cold Start).
    # We perform a dummy run to "warm up" the embedding model so 0% change isn't artificially penalized.
    # By using the full base text, we force LlamaIndex and PyTorch to allocate realistic buffers.
    logger.info("Initializing and warming up embedding model to prevent cold-start skew...")
    _ = benchmark_single_run(
        base, base, "Warmup", embedding_model_name=embedding_model
    )
    logger.info("Warm-up complete. Starting real benchmarks.")

    all_results = []

    for label, base_text, mod_text in scenarios:
        logger.info("=" * 60)
        logger.info("Benchmark: %s (averaging over %d runs)", label, num_runs)
        logger.info("=" * 60)

        run_results = []
        for run_i in range(num_runs):
            logger.info("  Run %d/%d", run_i + 1, num_runs)
            result = benchmark_single_run(
                base_text, mod_text, f"{label} (run {run_i+1})",
                embedding_model_name=embedding_model,
            )
            run_results.append(result)

        # Average the numeric fields
        avg = {"label": label}
        numeric_keys = [
            k for k in run_results[0]
            if isinstance(run_results[0][k], (int, float)) and k != "embed_speedup"
        ]
        for k in numeric_keys:
            values = [r[k] for r in run_results]
            avg[k] = round(sum(values) / len(values), 4)

        # Recompute speedup from averages
        if avg.get("time_embed_incremental_s", 0) > 0:
            avg["embed_speedup"] = round(
                avg["time_embed_full_s"] / avg["time_embed_incremental_s"], 2
            )
        else:
            avg["embed_speedup"] = float("inf")

        avg["num_runs"] = num_runs
        all_results.append(avg)

        logger.info(
            "  AVG: embed_incr=%.4fs, embed_full=%.4fs, speedup=%.2fx, "
            "efficiency=%.1f%%",
            avg.get("time_embed_incremental_s", 0),
            avg.get("time_embed_full_s", 0),
            avg.get("embed_speedup", 0),
            avg.get("efficiency_pct", 0),
        )

    return {
        "system_info": get_system_info(),
        "results": all_results,
    }


def print_results_table(results: list[dict]) -> None:
    """Print human-readable results table."""
    print("\n" + "=" * 100)
    print("COMPUTATIONAL PERFORMANCE BENCHMARK — Incremental vs Full Embedding")
    print("=" * 100)
    print(
        f"{'Scenario':<16} {'#Old':>5} {'#New':>5} "
        f"{'Unchg':>5} {'Mod':>4} {'Add':>4} "
        f"{'Embed(inc)':>11} {'Embed(full)':>12} "
        f"{'Speedup':>8} {'Eff%':>6} {'Mem(MB)':>8}"
    )
    print("-" * 100)
    for r in results:
        print(
            f"{r['label']:<16} "
            f"{r.get('chunks_old', 0):>5} "
            f"{r.get('chunks_new', 0):>5} "
            f"{r.get('unchanged', 0):>5} "
            f"{r.get('modified', 0):>4} "
            f"{r.get('added', 0):>4} "
            f"{r.get('time_embed_incremental_s', 0):>10.4f}s "
            f"{r.get('time_embed_full_s', 0):>11.4f}s "
            f"{r.get('embed_speedup', 0):>7.2f}x "
            f"{r.get('efficiency_pct', 0):>5.1f}% "
            f"{r.get('memory_delta_mb', 0):>7.1f}"
        )
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Computational benchmark: incremental vs full embedding"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./benchmark_results",
        help="Directory for output files",
    )
    parser.add_argument(
        "--test-dir", type=str,
        default=str(PROJECT_ROOT / "test_documents"),
        help="Directory containing test documents",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of runs to average over",
    )
    parser.add_argument(
        "--embedding-model", type=str,
        default="nomic-ai/nomic-embed-text-v1.5",
        help="Embedding model to use",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = run_all_benchmarks(
        test_dir=Path(args.test_dir),
        num_runs=args.runs,
        embedding_model=args.embedding_model,
    )

    # Print table
    print_results_table(data["results"])

    # Print system info
    print("\nSystem Info:")
    for k, v in data["system_info"].items():
        print(f"  {k}: {v}")

    # Export JSON
    json_path = output_dir / "compute_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        # Handle inf values
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info("JSON results: %s", json_path)

    # Export CSV
    csv_path = output_dir / "compute_benchmark.csv"
    if data["results"]:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data["results"][0].keys())
            writer.writeheader()
            writer.writerows(data["results"])
        logger.info("CSV results: %s", csv_path)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
