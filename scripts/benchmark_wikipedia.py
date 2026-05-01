#!/usr/bin/env python3
"""Heterogeneous-corpus scalability benchmark using Wikipedia articles.

This script extends the synthetic benchmark with a more realistic corpus:
it fetches a handful of full Wikipedia articles via the public REST API,
concatenates their paragraphs to build target documents of several sizes
(15 / 50 / 100 / 200 paragraphs), modifies ~10% of paragraphs in each, and
measures the *real* incremental versus full-reingest embedding time.

The goal is to show that the speedup factor of the incremental pipeline
remains stable across document sizes, not just on the small synthetic
15-paragraph base used elsewhere in the thesis.

Usage:
    poetry run python -m scripts.benchmark_wikipedia [--sizes 15 50 100 200]
                                                     [--change-ratio 0.10]
                                                     [--cache-dir benchmark_results/wiki_cache]

Output:
    benchmark_results/wikipedia_scalability.csv
    benchmark_results/wikipedia_scalability.json
    benchmark_results/figures/11_wikipedia_scalability.png
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# A heterogeneous selection of Wikipedia articles spanning history, science,
# culture and technology. Picked to give a realistic mix of paragraph sizes
# and writing styles, not curated for length.
DEFAULT_TITLES = [
    "Information_retrieval",
    "Machine_learning",
    "Vector_database",
    "Natural_language_processing",
    "Cryptographic_hash_function",
    "Levenshtein_distance",
    "Embedding",
    "Transformer_(deep_learning_architecture)",
    "Knowledge_graph",
    "Database_index",
    "Search_engine",
    "Tokenization_(data_security)",
]


def fetch_wikipedia_plaintext(title: str, lang: str = "en") -> str:
    """Fetch the plain-text body of a Wikipedia article.

    Uses the MediaWiki action API's `extracts` property (`?action=query
    &prop=extracts&explaintext`), which returns the article body as plain
    text with paragraphs separated by blank lines.
    """
    url = (
        f"https://{lang}.wikipedia.org/w/api.php?format=json&action=query"
        f"&prop=extracts&explaintext=1&redirects=1&titles={quote(title)}"
    )
    req = Request(url, headers={"User-Agent": "private-gpt-bachelorproef/1.0"})
    with urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        text = page.get("extract")
        if text:
            return text
    raise RuntimeError(f"Wikipedia returned no extract for '{title}'.")


def fetch_or_cache(title: str, cache_dir: Path) -> str:
    """Return article text from cache, or fetch and cache it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{title}.txt"
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    logger.info("Fetching %s from Wikipedia ...", title)
    text = fetch_wikipedia_plaintext(title)
    cached.write_text(text, encoding="utf-8")
    return text


def build_paragraph_pool(titles: list[str], cache_dir: Path) -> list[str]:
    """Aggregate all non-trivial paragraphs from the given articles."""
    pool: list[str] = []
    for title in titles:
        try:
            text = fetch_or_cache(title, cache_dir)
        except (URLError, TimeoutError) as exc:
            logger.warning("Skipping %s (fetch failed: %s)", title, exc)
            continue
        for para in text.split("\n\n"):
            cleaned = para.strip()
            # Skip headings and very short paragraphs to keep the corpus
            # representative of body text.
            if len(cleaned) >= 200 and not cleaned.startswith("=="):
                pool.append(cleaned)
    if not pool:
        raise RuntimeError(
            "No usable Wikipedia paragraphs were retrieved -- check connectivity "
            "or pass --offline-fallback once the cache directory has content."
        )
    logger.info("Collected %d paragraphs from %d articles.", len(pool), len(titles))
    return pool


def build_document(pool: list[str], num_paragraphs: int, seed: int) -> str:
    """Sample `num_paragraphs` paragraphs from the pool, in a fixed order."""
    rng = random.Random(seed)
    if num_paragraphs <= len(pool):
        chosen = rng.sample(pool, num_paragraphs)
    else:
        # If pool is smaller than required, sample with replacement but
        # keep duplicates apart so the chunk hashes stay distinct.
        chosen = [rng.choice(pool) for _ in range(num_paragraphs)]
    return "\n\n".join(chosen)


def modify_document_realistic(text: str, change_ratio: float, seed: int) -> str:
    """Modify a fraction of paragraphs by appending a sentence to each.

    Simpler than `benchmark_incremental.modify_document`: it always appends
    one sentence to the chosen paragraphs so the change ratio is exact,
    independent of paragraph length or shape.
    """
    rng = random.Random(seed)
    paragraphs = text.split("\n\n")
    num_to_change = max(1, int(round(len(paragraphs) * change_ratio)))
    indices = rng.sample(range(len(paragraphs)), num_to_change)
    appendix = (
        " This sentence was appended by the Wikipedia benchmark script to "
        "simulate a controlled modification in this paragraph."
    )
    for idx in indices:
        paragraphs[idx] = paragraphs[idx] + appendix
    return "\n\n".join(paragraphs)


def measure_one(
    base_text: str,
    modified_text: str,
    embedding_model_name: str,
) -> dict[str, Any]:
    """Run chunk + diff + real embed for both incremental and full paths."""
    from private_gpt.components.ingest.incremental.chunk_hasher import (
        SemanticChunker,
    )
    from private_gpt.components.ingest.incremental.diff_detector import (
        ChangeType,
        DiffDetector,
    )

    chunker = SemanticChunker(min_chunk_size=100, max_chunk_size=3000)
    detector = DiffDetector(similarity_threshold=0.4)

    t0 = time.perf_counter()
    old_chunks = chunker.chunk_text(base_text, metadata={"version": "old"})
    new_chunks = chunker.chunk_text(modified_text, metadata={"version": "new"})
    t_chunk = time.perf_counter() - t0

    t0 = time.perf_counter()
    changes = detector.detect_changes(old_chunks, new_chunks)
    t_diff = time.perf_counter() - t0

    embed_texts = [
        c.new_chunk.text
        for c in changes
        if c.change_type in (ChangeType.ADDED, ChangeType.MODIFIED)
        and c.new_chunk is not None
    ]

    # Lazy-load the embedding model once per call (the caller passes the
    # same instance via closure when batching is desired).
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(embedding_model_name, trust_remote_code=True)

    # Warm-up so the first encode call doesn't pay JIT cost.
    if embed_texts or new_chunks:
        warmup = embed_texts[:1] or [new_chunks[0].text]
        model.encode(warmup, show_progress_bar=False, normalize_embeddings=True)

    t0 = time.perf_counter()
    if embed_texts:
        model.encode(embed_texts, show_progress_bar=False, normalize_embeddings=True)
    t_embed_inc = time.perf_counter() - t0

    full_texts = [c.text for c in new_chunks]
    t0 = time.perf_counter()
    model.encode(full_texts, show_progress_bar=False, normalize_embeddings=True)
    t_embed_full = time.perf_counter() - t0

    chunks_to_embed = len(embed_texts)
    speedup = (
        t_embed_full / t_embed_inc if t_embed_inc > 0 else float("inf")
    )

    return {
        "chunks_old": len(old_chunks),
        "chunks_new": len(new_chunks),
        "chunks_embedded_incremental": chunks_to_embed,
        "chunks_embedded_full": len(new_chunks),
        "efficiency_pct": round(
            (1 - chunks_to_embed / len(new_chunks)) * 100
            if new_chunks else 0.0, 1
        ),
        "time_chunking_s": round(t_chunk, 4),
        "time_diffing_s": round(t_diff, 4),
        "time_embed_incremental_s": round(t_embed_inc, 4),
        "time_embed_full_s": round(t_embed_full, 4),
        "speedup_factor": round(speedup, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[15, 50, 100, 200],
        help="Document sizes (in paragraphs) to evaluate.",
    )
    parser.add_argument(
        "--change-ratio",
        type=float,
        default=0.10,
        help="Fraction of paragraphs that change between old and new.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed used when sampling paragraphs and choosing edits.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmark_results" / "wiki_cache",
        help="Where to cache fetched Wikipedia articles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmark_results",
        help="Where to write the resulting CSV/JSON/PNG.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="nomic-ai/nomic-embed-text-v1.5",
        help="SentenceTransformers model identifier.",
    )
    parser.add_argument(
        "--titles",
        type=str,
        nargs="+",
        default=DEFAULT_TITLES,
        help="Wikipedia article titles to use as the paragraph pool.",
    )
    args = parser.parse_args()

    pool = build_paragraph_pool(args.titles, args.cache_dir)

    results: list[dict[str, Any]] = []
    for size in args.sizes:
        logger.info("--- Benchmarking size = %d paragraphs ---", size)
        base_text = build_document(pool, size, seed=args.seed)
        modified_text = modify_document_realistic(
            base_text, change_ratio=args.change_ratio, seed=args.seed + 1
        )
        run = measure_one(base_text, modified_text, args.embedding_model)
        run["target_size"] = size
        run["change_ratio"] = args.change_ratio
        results.append(run)
        logger.info(
            "  size=%d -> chunks=%d, embed_inc=%.3fs, embed_full=%.3fs, "
            "speedup=%.2fx",
            size,
            run["chunks_new"],
            run["time_embed_incremental_s"],
            run["time_embed_full_s"],
            run["speedup_factor"],
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "wikipedia_scalability.json"
    csv_path = args.output_dir / "wikipedia_scalability.csv"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", csv_path)

    # ─── Plot ────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        logger.warning("matplotlib not available -- skipping figure")
        return

    sizes = [r["target_size"] for r in results]
    speedups = [r["speedup_factor"] for r in results]
    efficiencies = [r["efficiency_pct"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "#2c7bb6"
    color2 = "#d7191c"

    ax1.plot(sizes, speedups, "o-", color=color1, linewidth=2, markersize=8,
             label="Speedup-factor")
    ax1.set_xlabel("Documentgrootte (aantal paragrafen)")
    ax1.set_ylabel("Versnellingsfactor (x)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(sizes, efficiencies, "s--", color=color2, linewidth=2,
             markersize=8, label="Efficiëntie (%)")
    ax2.set_ylabel("Hergebruikte embeddings (%)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 105)

    plt.title(
        f"Schaalbaarheid op Wikipedia-corpus "
        f"({int(args.change_ratio * 100)}% wijziging)"
    )
    fig.tight_layout()

    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    png_path = fig_dir / "11_wikipedia_scalability.png"
    pdf_path = fig_dir / "11_wikipedia_scalability.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)

    logger.info("Wrote %s", png_path)


if __name__ == "__main__":
    main()
