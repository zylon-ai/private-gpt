#!/usr/bin/env python3
"""RAGAS evaluation benchmark: Semantic Chunking vs Fixed-Size Chunking.

This script implements the A/B test described in the thesis methodology:
it compares retrieval quality between semantic chunking (from the PoC)
and fixed-size chunking (LlamaIndex default) using the RAGAS framework.

Metrics evaluated:
  - Context Recall:    Do the retrieved chunks cover all relevant information?
  - Context Precision: Are the most relevant chunks ranked highest?
  - Faithfulness:      Is the generated answer grounded in the retrieved context?

The evaluation uses a Golden Dataset of question/answer/context triples
stored in benchmark_results/golden_dataset.json.

Usage:
    pip install ragas datasets  # one-time setup
    python -m scripts.benchmark_ragas [--output-dir ./benchmark_results]

Note: This script can run standalone without the full PrivateGPT server.
It only requires the SemanticChunker component and an embedding model.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

# Add the project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Sample corpus (same paragraphs shared across all benchmarks) ──────

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


# ─── Chunking strategies ───────────────────────────────────────────────


def chunk_semantic(corpus: str) -> list[str]:
    """Chunk using the PoC SemanticChunker."""
    from private_gpt.components.ingest.incremental.chunk_hasher import (
        SemanticChunker,
    )

    chunker = SemanticChunker(min_chunk_size=100, max_chunk_size=3000)
    hashed_chunks = chunker.chunk_text(corpus, metadata={"strategy": "semantic"})
    return [c.text for c in hashed_chunks]


def chunk_fixed_size(corpus: str, chunk_size: int = 512, overlap: int = 20) -> list[str]:
    """Chunk using a simple fixed-size splitter (simulates LlamaIndex default).

    Uses a character-level sliding window with sentence-boundary awareness.
    """
    chunks: list[str] = []
    sentences = corpus.replace("\n\n", "\n").split(". ")
    current_chunk: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sent_with_period = sentence if sentence.endswith(".") else sentence + "."
        sent_len = len(sent_with_period)

        if current_len + sent_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep overlap by retaining the last sentence(s)
            overlap_text = ""
            keep: list[str] = []
            for s in reversed(current_chunk):
                if len(overlap_text) + len(s) <= overlap:
                    keep.insert(0, s)
                    overlap_text += s
                else:
                    break
            current_chunk = keep
            current_len = len(overlap_text)

        current_chunk.append(sent_with_period)
        current_len += sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ─── Simple embedding-based retriever ──────────────────────────────────


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SimpleRetriever:
    """A lightweight in-memory retriever using HuggingFace embeddings."""

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
            raise

        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name, trust_remote_code=True)
        self._chunks: list[str] = []
        self._embeddings: list[list[float]] = []

    def index(self, chunks: list[str]) -> None:
        """Index a list of text chunks."""
        self._chunks = chunks
        self._embeddings = self._model.encode(
            chunks, show_progress_bar=False, normalize_embeddings=True
        ).tolist()

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieve top-k most similar chunks for a query."""
        import itertools

        query_emb = self._model.encode(
            [query], normalize_embeddings=True
        ).tolist()[0]

        scored = [
            (cosine_similarity(query_emb, chunk_emb), chunk_text)
            for chunk_emb, chunk_text in zip(self._embeddings, self._chunks)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in itertools.islice(scored, top_k)]


# ─── Ollama LLM answer generation ────────────────────────────────────


def _check_ollama_available(url: str) -> bool:
    """Return True if the Ollama server at *url* is reachable."""
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.urlopen(url.rstrip("/") + "/", timeout=3)
        return req.status == 200
    except Exception:
        return False


def generate_answers_with_ollama(
    questions: list[str],
    retrieved_contexts: list[list[str]],
    url: str,
    model: str,
) -> list[str]:
    """Generate answers by prompting Ollama with each question + retrieved context.

    Each answer is produced by sending a single-turn prompt:
        Context: <retrieved chunks joined>
        Question: <question>
    to the Ollama /api/generate endpoint.
    """
    import json as _json
    import urllib.request

    answers: list[str] = []
    api_url = url.rstrip("/") + "/api/generate"

    for i, (q, ctxs) in enumerate(zip(questions, retrieved_contexts)):
        context_text = "\n\n".join(ctxs)
        prompt = (
            f"Answer the following question using only the provided context.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {q}\n\nAnswer:"
        )
        payload = _json.dumps(
            {"model": model, "prompt": prompt, "stream": False}
        ).encode()
        try:
            req = urllib.request.Request(
                api_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = _json.loads(resp.read())
                answer = body.get("response", "").strip()
        except Exception as exc:
            logger.warning("Ollama request %d/%d failed: %s", i + 1, len(questions), exc)
            answer = ""
        answers.append(answer)
        logger.info("  [%d/%d] Generated answer (%d chars)", i + 1, len(questions), len(answer))

    return answers


# ─── RAGAS evaluation ─────────────────────────────────────────────────


def evaluate_with_ragas(
    questions: list[str],
    ground_truths: list[str],
    retrieved_contexts: list[list[str]],
    answers: list[str] | None,
) -> dict:
    """Run RAGAS evaluation and return metric scores.

    *answers* may be None when no LLM is available; in that case faithfulness
    is omitted from the results and set to None.

    Returns a dict with per-question scores and averages.
    """
    skip_faithfulness = answers is None
    # Provide a dummy answers list so RAGAS / manual fallback can run without it
    effective_answers = answers if answers is not None else ground_truths

    try:
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            context_precision,
            context_recall,
        )
        if not skip_faithfulness:
            from ragas.metrics import faithfulness as _faithfulness  # type: ignore
            metrics = [context_recall, context_precision, _faithfulness]
        else:
            metrics = [context_recall, context_precision]
    except ImportError:
        logger.warning(
            "RAGAS not installed. Falling back to manual context overlap scoring.\n"
            "Install with: pip install ragas datasets"
        )
        return _evaluate_manual(
            questions, ground_truths, retrieved_contexts, effective_answers,
            skip_faithfulness=skip_faithfulness,
        )

    # Build the RAGAS dataset
    data = {
        "question": questions,
        "answer": effective_answers,
        "contexts": retrieved_contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    logger.info("Running RAGAS evaluation with %d samples...", len(questions))
    try:
        result = evaluate(dataset, metrics=metrics)
        out = {
            "context_recall": float(result["context_recall"]),
            "context_precision": float(result["context_precision"]),
            "faithfulness": float(result["faithfulness"]) if not skip_faithfulness else None,
            "per_question": result.to_pandas().to_dict("records") if hasattr(result, "to_pandas") else [],
        }
        return out
    except Exception as e:
        logger.warning("RAGAS evaluation failed (%s), using manual fallback.", e)
        return _evaluate_manual(
            questions, ground_truths, retrieved_contexts, effective_answers,
            skip_faithfulness=skip_faithfulness,
        )


def _evaluate_manual(
    questions: list[str],
    ground_truths: list[str],
    retrieved_contexts: list[list[str]],
    answers: list[str],
    skip_faithfulness: bool = False,
) -> dict:
    """Manual context overlap evaluation as fallback when RAGAS is unavailable.

    Computes:
    - Context Recall:    fraction of ground-truth tokens found in retrieved contexts
    - Context Precision: fraction of retrieved context tokens found in ground-truth
    - Faithfulness:      token overlap between answer and ground-truth (proxy),
                         only when skip_faithfulness=False
    """
    import re

    def tokenize(text: str) -> set[str]:
        return set(re.findall(r'\w+', text.lower()))

    recalls, precisions, relevances = [], [], []

    for gt, ctxs, ans in zip(ground_truths, retrieved_contexts, answers):
        gt_tokens = tokenize(gt)
        ctx_text = " ".join(ctxs)
        ctx_tokens = tokenize(ctx_text)

        # Context Recall: how much of ground truth is covered by contexts
        if gt_tokens:
            recall = len(gt_tokens & ctx_tokens) / len(gt_tokens)
        else:
            recall = 1.0
        recalls.append(recall)

        # Context Precision: how focused are the contexts on the ground truth
        if ctx_tokens:
            precision = len(gt_tokens & ctx_tokens) / len(ctx_tokens)
        else:
            precision = 0.0
        precisions.append(precision)

        if not skip_faithfulness:
            ans_tokens = tokenize(ans)
            if gt_tokens:
                relevance = len(gt_tokens & ans_tokens) / len(gt_tokens)
            else:
                relevance = 1.0
            relevances.append(relevance)

    faithfulness_score: float | None = (
        sum(relevances) / len(relevances) if relevances else None
    )

    return {
        "context_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "context_precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "faithfulness": faithfulness_score,
        "per_question": [
            {
                "question": q,
                "context_recall": r,
                "context_precision": p,
            }
            for q, r, p in zip(questions, recalls, precisions)
        ],
    }


# ─── Main benchmark runner ────────────────────────────────────────────


def run_benchmark(
    golden_dataset_path: Path,
    output_dir: Path,
    top_k: int = 3,
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
    ollama_url: str | None = None,
    ollama_model: str = "llama3.1",
) -> dict:
    """Run the full A/B benchmark comparing semantic vs fixed chunking.

    When *ollama_url* is provided and the Ollama server is reachable, real
    LLM answers are generated for each question so faithfulness can be
    evaluated meaningfully.  Otherwise faithfulness is omitted from results.
    """
    # Load Golden Dataset
    with open(golden_dataset_path, encoding="utf-8") as f:
        golden_data = json.load(f)

    questions = [item["question"] for item in golden_data]
    ground_truths = [item["ground_truth"] for item in golden_data]

    # Determine whether Ollama is available for faithfulness evaluation
    ollama_available = False
    if ollama_url:
        ollama_available = _check_ollama_available(ollama_url)
        if ollama_available:
            logger.info(
                "Ollama reachable at %s (model: %s) — faithfulness will be evaluated.",
                ollama_url, ollama_model,
            )
        else:
            logger.warning(
                "Ollama not reachable at %s — faithfulness will be skipped.",
                ollama_url,
            )
    else:
        logger.info(
            "No --ollama-url provided — faithfulness will be skipped. "
            "Pass --ollama-url http://localhost:11434 to enable LLM-based evaluation."
        )

    # Build the full corpus
    corpus = "\n\n".join(SAMPLE_PARAGRAPHS)

    # Initialize retriever
    retriever = SimpleRetriever(model_name=embedding_model)

    results = {}

    for strategy_name, chunk_fn in [
        ("Semantic Chunking", lambda: chunk_semantic(corpus)),
        ("Fixed-Size Chunking (512)", lambda: chunk_fixed_size(corpus, chunk_size=512, overlap=20)),
        ("Fixed-Size Chunking (1024)", lambda: chunk_fixed_size(corpus, chunk_size=1024, overlap=20)),
    ]:
        logger.info("=" * 60)
        logger.info("Evaluating: %s", strategy_name)
        logger.info("=" * 60)

        # Chunk
        t0 = time.perf_counter()
        chunks = chunk_fn()
        chunk_time = time.perf_counter() - t0
        logger.info("  Chunked into %d chunks (%.4fs)", len(chunks), chunk_time)

        # Index
        t0 = time.perf_counter()
        retriever.index(chunks)
        index_time = time.perf_counter() - t0
        logger.info("  Indexed %d chunks (%.4fs)", len(chunks), index_time)

        # Retrieve contexts for each question
        all_retrieved: list[list[str]] = []
        for q in questions:
            retrieved = retriever.retrieve(q, top_k=top_k)
            all_retrieved.append(retrieved)

        # Generate answers via Ollama (enables real faithfulness evaluation),
        # or pass None to skip faithfulness.
        if ollama_available and ollama_url:
            logger.info("  Generating answers via Ollama (%s)...", ollama_model)
            answers: list[str] | None = generate_answers_with_ollama(
                questions, all_retrieved, ollama_url, ollama_model
            )
        else:
            answers = None

        # Evaluate
        eval_result = evaluate_with_ragas(
            questions=questions,
            ground_truths=ground_truths,
            retrieved_contexts=all_retrieved,
            answers=answers,
        )

        results[strategy_name] = {
            "num_chunks": len(chunks),
            "chunk_time_s": chunk_time,
            "index_time_s": index_time,
            **eval_result,
        }

        faithfulness_str = (
            f"{eval_result['faithfulness']:.3f}"
            if eval_result["faithfulness"] is not None
            else "N/A"
        )
        logger.info(
            "  Results: recall=%.3f, precision=%.3f, faithfulness=%s",
            eval_result["context_recall"],
            eval_result["context_precision"],
            faithfulness_str,
        )

    return results


def print_comparison_table(results: dict) -> None:
    """Print a formatted comparison table."""
    any_faithfulness = any(
        v.get("faithfulness") is not None for v in results.values()
    )
    print("\n" + "=" * 85)
    print("RAGAS EVALUATION: Semantic vs Fixed-Size Chunking")
    if not any_faithfulness:
        print(
            "NOTE: Faithfulness N/A -- pass --ollama-url to enable LLM-based evaluation."
        )
    print("=" * 85)
    print(
        f"{'Strategy':<30} {'#Chunks':>7} "
        f"{'Recall':>8} {'Precision':>10} {'Faithful':>10}"
    )
    print("-" * 85)
    for name, data in results.items():
        faith = data.get("faithfulness")
        faith_str = f"{faith:>10.3f}" if faith is not None else f"{'N/A':>10}"
        print(
            f"{name:<30} {data['num_chunks']:>7} "
            f"{data['context_recall']:>8.3f} "
            f"{data['context_precision']:>10.3f} "
            f"{faith_str}"
        )
    print("=" * 85)


def export_results(results: dict, output_dir: Path) -> None:
    """Export results as CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON export
    json_path = output_dir / "ragas_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results exported to %s", json_path)

    # CSV export (summary)
    csv_path = output_dir / "ragas_results.csv"
    rows = []
    for strategy, data in results.items():
        faith = data.get("faithfulness")
        rows.append({
            "strategy": strategy,
            "num_chunks": data["num_chunks"],
            "chunk_time_s": data["chunk_time_s"],
            "index_time_s": data["index_time_s"],
            "context_recall": data["context_recall"],
            "context_precision": data["context_precision"],
            "faithfulness": faith if faith is not None else "N/A",
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Summary CSV exported to %s", csv_path)

    # Per-question CSV export
    per_q_path = output_dir / "ragas_per_question.csv"
    per_q_rows = []
    for strategy, data in results.items():
        for pq in data.get("per_question", []):
            per_q_rows.append({"strategy": strategy, **pq})

    if per_q_rows:
        with open(per_q_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=per_q_rows[0].keys())
            writer.writeheader()
            writer.writerows(per_q_rows)
        logger.info("Per-question results exported to %s", per_q_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation: Semantic vs Fixed-Size Chunking"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Directory to store evaluation results",
    )
    parser.add_argument(
        "--golden-dataset",
        type=str,
        default=str(PROJECT_ROOT / "benchmark_results" / "golden_dataset.json"),
        help="Path to the Golden Dataset JSON file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve per question",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="nomic-ai/nomic-embed-text-v1.5",
        help="HuggingFace embedding model to use",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=None,
        help=(
            "Ollama server URL for LLM-based faithfulness evaluation "
            "(e.g. http://localhost:11434). If omitted, faithfulness is skipped."
        ),
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="llama3.1",
        help="Ollama model name to use for answer generation (default: llama3.1)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    golden_path = Path(args.golden_dataset)

    if not golden_path.exists():
        logger.error("Golden Dataset not found at %s", golden_path)
        sys.exit(1)

    results = run_benchmark(
        golden_dataset_path=golden_path,
        output_dir=output_dir,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
    )

    print_comparison_table(results)
    export_results(results, output_dir)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
