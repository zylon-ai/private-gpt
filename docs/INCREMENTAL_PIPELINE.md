# Incremental Update Pipeline — End-to-End Documentation

> Complete technical description of all components that together form the incremental update pipeline in PrivateGPT: from file detection to vector store update, including GUI integration, RAGAS evaluation, and manual testing guide.

## Two Independent Features

This PoC adds **two orthogonal capabilities**. They are often used together but are architecturally separate:

| Feature | What it is | Where it lives |
|---|---|---|
| **Semantic chunking** | A *chunking strategy* — splits text on paragraph/heading boundaries instead of fixed token windows | `chunk_hasher.py` (`SemanticChunker`) |
| **Incremental updates** | An *update pipeline* — compares content hashes and re-embeds only changed chunks instead of re-embedding the whole document | `incremental_updater.py`, `diff_detector.py`, `chunk_hash_store.py` |

Semantic chunking improves retrieval quality (RAGAS benchmark) regardless of whether updates are incremental. Incremental updates save compute regardless of which chunker is used — but they work *best* with a stable chunker like the semantic one, because fixed-size chunking suffers from the avalanche effect (a one-paragraph edit shifts every subsequent chunk boundary, forcing a full re-embed).

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Configuration (settings.yaml)](#2-configuration)
3. [Semantic Chunker](#3-semantic-chunker)
4. [Chunk Hash Store](#4-chunk-hash-store)
5. [Diff Detector](#5-diff-detector)
6. [Incremental Updater](#6-incremental-updater)
7. [Ingest Component (PrivateGPT Integration)](#7-ingest-component)
8. [Incremental Watcher (File Watching)](#8-incremental-watcher)
9. [GUI Integration (Gradio UI)](#9-gui-integration)
10. [RAGAS Evaluation](#10-ragas-evaluation)
11. [Computational Benchmarks](#11-computational-benchmarks)
12. [Manual Testing Guide](#12-manual-testing-guide)
13. [File Overview](#13-file-overview)

---

## 1. Architecture Overview

The pipeline replaces PrivateGPT's default "delete-and-re-ingest" behaviour with an intelligent system that **only re-embeds changed chunks**. This prevents the avalanche effect of fixed-size chunking and saves computational work.

### End-to-End Flow

```
Document (new or modified)
    |
    v
[1] SemanticChunker  (chunk_hasher.py)
    Splits on paragraph boundaries instead of fixed token windows
    |
    v
[2] HashedChunks  (text + SHA-256 per chunk)
    |
    v
[3] DiffDetector  (diff_detector.py)  <--- ChunkHashStore (stored hashes)
    Compares new hashes with stored hashes
    |
    v
[4] Change Classification
    |--- UNCHANGED ---> Skip (no re-embedding)
    |--- MODIFIED  ---> Delete old node + embed new chunk
    |--- ADDED     ---> Embed new chunk + insert into index
    |--- DELETED   ---> Delete node from index
    |
    v
[5] VectorStoreIndex  (LlamaIndex)
    insert_nodes() / delete_nodes()
    |
    v
[6] ChunkHashStore  (update registry for future diffs)
    |
    v
[7] Persist index + registry to disk
```

### Entry Points

| Entry Point | Component | Triggers |
|---|---|---|
| **User upload via UI** | `IncrementalIngestComponent` | Upload button in Gradio sidebar |
| **File watcher** | `IncrementalIngestWatcher` | Automatic detection of file changes |
| **API call** | `IngestService` | REST API `/v1/ingest/file` |

---

## 2. Configuration

**File**: `settings.yaml`
**Model**: `IncrementalSettings` in `private_gpt/settings/settings.py`

```yaml
embedding:
  mode: huggingface
  ingest_mode: incremental        # one way to activate the incremental pipeline
  embed_dim: 768

incremental:
  enabled: true                   # other way to activate (either is sufficient)
  min_chunk_size: 100             # minimum characters per chunk
  max_chunk_size: 3000            # maximum characters per chunk
  similarity_threshold: 0.4       # Ratcliff/Obershelp threshold
  debounce_seconds: 2.0           # debounce for file watcher
```

### Activating the incremental pipeline

The factory in `get_ingestion_component()` selects `IncrementalIngestComponent` when **either**:
- `embedding.ingest_mode == "incremental"`, **or**
- `incremental.enabled == true`

Having two knobs keeps backwards compatibility with the original `ingest_mode` field while making `incremental.enabled` the more discoverable primary toggle.

| Parameter | Default | Description |
|---|---|---|
| `incremental.enabled` | `true` | Primary toggle — selects `IncrementalIngestComponent` in the factory |
| `ingest_mode` | `incremental` | Alternative toggle — same effect as `incremental.enabled` |
| `min_chunk_size` | `100` | Chunks smaller than this are merged with the next paragraph |
| `max_chunk_size` | `3000` | Chunks larger than this are split at sentence boundaries |
| `similarity_threshold` | `0.4` | Minimum similarity to consider two chunks as "modified" vs "one deleted + one added" |
| `debounce_seconds` | `2.0` | Minimum interval between watcher events for the same file |

**Note on scope:** these settings control the *incremental update pipeline*. The *semantic chunker* is always used when the incremental pipeline is active — it is not independently toggleable, because the diff algorithm relies on the chunker's stable boundaries.

---

## 3. Semantic Chunker

**File**: `private_gpt/components/ingest/incremental/chunk_hasher.py` (215 lines)

### Purpose
Splits text on **semantic boundaries** (paragraphs, headings, separators) instead of fixed token windows. This prevents the *avalanche effect* where a small edit shifts all subsequent chunks.

### Classes

**`HashedChunk`** (dataclass):
- `chunk_index`: Position in the document (0-based)
- `text`: Raw chunk text
- `content_hash`: SHA-256 of the normalised text
- `metadata`: Arbitrary metadata (file_name, chunk_index, etc.)

**`SemanticChunker`**:
```python
chunker = SemanticChunker(min_chunk_size=100, max_chunk_size=3000)
chunks = chunker.chunk_text(full_text, metadata={"file_name": "report.pdf"})
```

### Algorithm

1. **Split on paragraph boundaries** via regex:
   - Double newline (`\n\n`)
   - Markdown headings (`# Title`)
   - Separators (`---`, `===`)

2. **Merge small chunks** (< `min_chunk_size`): merged with the following paragraph

3. **Split oversized chunks** (> `max_chunk_size`): split at sentence boundaries

4. **Hash computation**: SHA-256 of normalised text (whitespace collapsed, so insignificant whitespace changes don't trigger re-embedding)

---

## 4. Chunk Hash Store

**File**: `private_gpt/components/ingest/incremental/chunk_hash_store.py` (185 lines)

### Purpose
Persistent JSON registry that stores chunk hashes, node IDs, and full text for each document. Enables fast change detection on subsequent ingestions.

### Data Structures

**`StoredChunkInfo`**:
- `chunk_index`: Position in document
- `content_hash`: SHA-256 hash
- `node_id`: LlamaIndex node ID (for delete/update operations)
- `text_preview`: First 100 characters (for debugging)
- `full_text`: Complete text (for SequenceMatcher comparison)

**`DocumentRecord`**:
- `doc_id`: Unique document ID
- `file_name`: Original filename
- `file_hash`: SHA-256 of the complete file
- `chunks`: List of `StoredChunkInfo`
- `version`: Monotonically increasing version number

### Storage

```
local_data/
  chunk_hash_registry.json    <-- persistent registry
```

### API

| Method | Description |
|---|---|
| `get_document(doc_id)` | Lookup record by doc_id |
| `get_document_by_filename(name)` | Lookup record by filename |
| `upsert_document(record)` | Insert or update + persist to disk |
| `delete_document(doc_id)` | Remove record + persist to disk |
| `get_chunk_hashes(doc_id)` | Returns `{chunk_index: hash}` mapping |
| `get_chunk_node_ids(doc_id)` | Returns `{chunk_index: node_id}` mapping |

Thread-safe via `threading.RLock`.

---

## 5. Diff Detector

**File**: `private_gpt/components/ingest/incremental/diff_detector.py` (422 lines)

### Purpose
Compares old and new chunk lists to determine which chunks are ADDED, MODIFIED, DELETED, or UNCHANGED.

### ChangeType Enum
- `ADDED` — New chunk, must be embedded
- `MODIFIED` — Existing chunk changed, re-embed required
- `DELETED` — Chunk removed, delete from index
- `UNCHANGED` — Identical, skip

### Three-Phase Algorithm

**Phase 1 — Hash Matching (O(n))**:
- Build `hash -> [chunks]` mappings for old and new
- Match chunks with identical hashes via *proximity pairing* (minimise |old_index − new_index|)
- Result: UNCHANGED chunks

**Phase 2 — Sequence Matching (Ratcliff/Obershelp)**:
- For unmatched chunks: compute `SequenceMatcher.ratio()`
- If ratio >= `similarity_threshold` (default 0.4) → MODIFIED
- Greedy best-match pairing

**Phase 3 — Remaining**:
- Unmatched old chunks → DELETED
- Unmatched new chunks → ADDED

### Patience Diff (bonus)
A standalone `patience_diff()` function implementing the Patience Diff algorithm:
1. Find lines that appear **exactly once** in both versions (anchor points)
2. Compute LIS (Longest Increasing Subsequence) on anchor positions
3. Apply LCS to segments between anchors

---

## 6. Incremental Updater

**File**: `private_gpt/components/ingest/incremental/incremental_updater.py` (584 lines)

### Purpose
The **core orchestrator** of the pipeline. Coordinates all steps from file reading to vector store update.

### `ingest_file()` — The Complete Process

```
Step 1:  Read file via IngestionHelper
Step 2:  Compute file_hash (SHA-256 of full text)
          -> File unchanged? SKIP (return immediately)
Step 3:  Chunk with SemanticChunker
          -> New file? _full_ingest() (embed everything)
Step 4:  Build old chunks from ChunkHashStore
Step 5:  DiffDetector.detect_changes(old, new)
Step 6:  Delete DELETED + MODIFIED nodes from index
Step 7:  Create TextNodes for ADDED + MODIFIED chunks
Step 8:  run_transformations() (compute embeddings)
Step 9:  index.insert_nodes() (add to vector store)
Step 10: Update ChunkHashStore with new hashes + node IDs
Step 11: Persist index + registry
```

### `IncrementalUpdateStats` — Performance Metrics

| Field | Description |
|---|---|
| `chunks_unchanged` | Chunks that were not re-embedded |
| `chunks_modified` | Chunks that were changed and re-embedded |
| `chunks_added` | New chunks that were embedded |
| `chunks_deleted` | Chunks removed from the index |
| `time_chunking_s` | Time spent on chunking |
| `time_diffing_s` | Time spent on diff detection |
| `time_embedding_s` | Time spent computing embeddings |
| `time_indexing_s` | Time spent updating the vector store |
| `efficiency_ratio` | Fraction of skipped embeddings (0.8 = 80% reuse) |

---

## 7. Ingest Component

**File**: `private_gpt/components/ingest/incremental/ingest_component.py` (197 lines)

### Purpose
**Bridge** between the standalone `IncrementalUpdater` and PrivateGPT's dependency-injected `IngestService`. Implements `BaseIngestComponentWithIndex` so the rest of the application (API, UI, CLI) uses the incremental pipeline transparently.

### Factory Selection

In the main `ingest_component.py`, the `get_ingestion_component()` factory checks `settings.embedding.ingest_mode`:
- `"incremental"` → `IncrementalIngestComponent`
- `"simple"` → `SimpleIngestComponent`

### Transform Filtering
The `IncrementalIngestComponent` filters the standard transformations and **keeps only embedding transforms**:
```python
embed_only_transforms = [t for t in transformations if isinstance(t, BaseEmbedding)]
```
This prevents LlamaIndex's `SentenceWindowNodeParser` from re-splitting the semantic chunks.

### Interface Methods

| Method | Behaviour |
|---|---|
| `ingest(file_name, file_data)` | Incremental: only re-embed changed chunks |
| `bulk_ingest(files)` | Sequential per file (savings come from chunk reuse) |
| `delete(doc_id)` | Remove via hash store or fallback to ref_doc delete |

---

## 8. Incremental Watcher

**File**: `private_gpt/components/ingest/incremental/incremental_watcher.py`

### Purpose
Watches **specific registered files** on disk and triggers the incremental pipeline when any of them is modified or deleted. Uses `watchdog` under the hood, but schedules handlers scoped per parent directory so only the registered files fire — unrelated neighbours are ignored.

This is a **per-file** watcher, not a directory watcher. The user registers each file path they want watched (via the UI's *Watch file by path* input or `ingest_file_from_path()` in the service), and the list is persisted to `local_data/watched_files.json` so it survives restarts.

### Features
- **Per-file registration**: `add_file_watch(path, on_modified, on_deleted)`
- **Debouncing**: Suppresses rapid-fire events (editors that write multiple times)
- **Pre-arm (`touch_debounce`)**: Call before a programmatic write to stop the watcher from re-firing on our own upload copy
- **Persistence**: Registered paths survive server restarts
- **Two event types**: `modified` → incremental update, `deleted` → remove from index

---

## 9. GUI Integration

**File**: `private_gpt/ui/ui.py`

### Purpose
Surfaces the incremental pipeline's status and the per-file watcher in the Gradio UI. **The mode is not a runtime toggle** — it is read from `settings.yaml` at startup (because `IngestService` is a singleton and its ingest component is wired at construction time). Flipping a checkbox at runtime could not safely re-wire the singleton, so the UI reflects the configured state instead of trying to mutate it.

### Components
- **Static mode indicator** (`gr.Markdown`): shows either
  - `"Incremental mode — only modified chunks are re-embedded"` if `incremental.enabled` or `ingest_mode == "incremental"`, or
  - `"Standard mode — files are fully re-processed on every upload"` otherwise.
  Configuration instructions direct the user to `settings.yaml`.
- **File Watcher block**:
  - Start / Stop buttons for the watchdog observer thread
  - **Watch file by path (optional)** — paste an absolute path to a file on disk. Use this only if you want to watch a file *in-place* on your own filesystem rather than the copy under `local_data/uploads/`.
  - **Unwatch all** — clears the registered list
  - Live status text showing watcher state and currently-registered files
- **Upload button behaviour (incremental mode)**:
  1. Pre-arm `touch_debounce` so the watcher's own filesystem event is suppressed
  2. Copy the Gradio tmp file into `local_data/uploads/<filename>` (stable path — Gradio tmp paths are deleted after the request)
  3. Auto-start the watcher thread if it isn't running
  4. Call `ingest_file_from_path()` which does the incremental diff-and-re-embed AND registers the persistent path with the watcher
  After upload, editing or re-saving `local_data/uploads/<filename>` fires the watcher automatically — no manual registration needed.

**Why `uploads/` exists**: browser uploads give Gradio a tmp path that is cleaned up after the request ends. The watcher needs a stable path to monitor, so we copy the upload into `local_data/uploads/`. There is no way for the browser to expose the client's original filesystem path (browser security), so "watch the user's original file" requires the explicit *Watch file by path* input.

---

## 10. RAGAS Evaluation

**Script**: `scripts/benchmark_ragas.py`
**Dataset**: `benchmark_results/golden_dataset.json`

### Purpose
Quantitative A/B test comparing **retrieval quality** of semantic chunking vs fixed-size chunking using the RAGAS framework.

### Metrics

| Metric | What It Measures |
|---|---|
| **Context Recall** | Do the retrieved chunks cover all relevant information? |
| **Context Precision** | Are the most relevant chunks ranked highest (top-k)? |
| **Faithfulness** | Is the generated answer grounded in the retrieved context? |

### Results

| Strategy | #Chunks | Context Recall | Context Precision | Faithfulness |
|---|---|---|---|---|
| **Semantic Chunking** | **15** | **0.970** | **0.356** | **1.000** |
| Fixed-Size (512) | 10 | 0.939 | 0.240 | 1.000 |
| Fixed-Size (1024) | 4 | 0.932 | 0.115 | 1.000 |

**Conclusion**: Semantic chunking achieves the highest Context Recall (+3–4%) and Context Precision (+48–209%), demonstrating that smaller, semantically coherent chunks improve retrieval quality.

---

## 11. Computational Benchmarks

### Benchmark Scripts

| Script | What It Measures |
|---|---|
| `benchmark_compute.py` | **Real** wall-clock embedding time for the incremental pipeline vs full re-embed. Use this for defensible performance numbers. |
| `benchmark_incremental.py` | Speedup factor and embedding reuse ratio for different change levels — **note: embedding times are estimated (50 ms/chunk)**; run `benchmark_compute.py` for real measurements. |
| `benchmark_ragas.py` | Retrieval quality (Context Recall, Context Precision) comparing the **semantic chunker** vs **fixed-size chunking**. Pass `--ollama-url` to enable LLM faithfulness. This measures the *chunking strategy*, independent of the incremental update pipeline. |
| `benchmark_quality.py` | Correctness of the incremental pipeline: context drift on unchanged chunks, diff accuracy F1, hash stability, scalability. |
| `benchmark_quality_full.py` | Same three-panel correctness report applied to the **baseline full re-ingest pipeline** (fixed-size chunker, no hash store) — exposes inefficiency and avalanche effect. Pair with `benchmark_quality.py` for side-by-side comparison. |

### Running the Computational Benchmark

```bash
# Basic run (3 repetitions)
poetry run python -m scripts.benchmark_compute --runs 3

# More repetitions for statistical significance
poetry run python -m scripts.benchmark_compute --runs 10

# Use a different embedding model
poetry run python -m scripts.benchmark_compute --embedding-model all-MiniLM-L6-v2
```

### What `benchmark_compute.py` Measures

| Metric | Description |
|---|---|
| `time_embed_incremental_s` | Real wall-clock time for embedding only changed chunks |
| `time_embed_full_s` | Real wall-clock time for embedding all chunks (baseline) |
| `embed_speedup` | Ratio: full / incremental (e.g. 5.0x = 5 times faster) |
| `time_cpu_total_s` | Total CPU time (user + system) |
| `memory_before_mb` | RSS before benchmark |
| `memory_after_mb` | RSS after benchmark |
| `memory_delta_mb` | Memory consumed by the operation |
| `efficiency_pct` | Percentage of embeddings that were skipped |

### Tips for Best Measurement Results

1. **Use a VM or dedicated machine** — No background processes competing for CPU/memory
2. **Close unnecessary applications** — Especially browsers and IDEs
3. **Run multiple repetitions** — `--runs 10` averages out variance
4. **Install psutil** for accurate memory measurement: `pip install psutil`
5. **Disable turbo boost** on the CPU if you want stable clock speeds

---

## 12. Manual Testing Guide

This section explains how to manually test the incremental update pipeline step by step using the provided test documents.

### Prerequisites

1. In `settings.yaml`, set either `incremental.enabled: true` or `embedding.ingest_mode: incremental`
2. Start PrivateGPT: `poetry run python -m private_gpt`
3. Open the Gradio UI at `http://localhost:8001` — the sidebar should show `"Incremental mode — only modified chunks are re-embedded"`
4. The test documents are in `test_documents/`:
   - `base_document.txt` — Original document (15 paragraphs, 0% change)
   - `modified_10pct.txt` — 10% change (1 paragraph has an added sentence)
   - `modified_50pct.txt` — 50% change (7 paragraphs modified/replaced)
   - `modified_90pct.txt` — 90% change (13 paragraphs completely replaced)

### Test Procedure

**Test 1: Initial Ingestion**
1. Upload `base_document.txt` via the UI
2. Check the server logs for `IncrementalUpdateStats`:
   - Expected: `chunks_added: 15`, `chunks_unchanged: 0`, `efficiency_ratio: 0.0%`
   - This is a new file, so all chunks are embedded
3. Ask a question: *"What is the avalanche effect in chunking?"*
4. Verify the answer references the correct paragraph

**Test 2: No Change (0% modification)**
1. Upload `base_document.txt` again (same file, same name)
2. Check server logs:
   - Expected: `"File unchanged (same file hash). Skipping."`
   - No embeddings should be computed
   - This proves the file-level skip detection works

**Test 3: Small Change (10%)**
1. Upload `modified_10pct.txt` (rename it to `base_document.txt` first, or delete the old one)
2. Check server logs:
   - Expected:  1 modified chunk,  14 unchanged
   - Expected: `efficiency_ratio:  93%` (only 1 out of 15 chunks re-embedded)
3. Ask: *"What does GDPR require for local processing?"*
4. Verify the answer picks up the newly added sentence

**Test 4: Medium Change (50%)**
1. Delete the existing document from the UI
2. Upload `base_document.txt` first (to establish baseline)
3. Then upload `modified_50pct.txt` (with the same filename)
4. Check server logs:
   - Expected:  7-8 modified/added chunks,  7-8 unchanged
   - Expected: `efficiency_ratio:  50%`

**Test 5: Large Change (90%)**
1. Repeat the same procedure with `modified_90pct.txt`
2. Check server logs:
   - Expected:  13 modified/added,  2 unchanged
   - Expected: `efficiency_ratio:  13%`
   - Even with 90% change, the 2 unchanged chunks are still skipped

### Interpreting Server Logs

When a file is ingested incrementally, the logs show:
```
Incremental update for 'base_document.txt':
  Chunks: 15 old -> 15 new
  Unchanged: 14 | Modified: 1 | Added: 0 | Deleted: 0
  Embeddings: 1 computed, 14 skipped (93.3% reuse)
  Timing: chunk=0.001s, diff=0.000s, embed=0.150s, index=0.050s, total=0.201s
```

Key fields to look at:
- **Unchanged**: How many chunks were skipped (higher = more efficient)
- **Embeddings computed vs skipped**: Shows real computational savings
- **Timing**: Shows where time is spent (chunking, diffing, embedding, indexing)

### Running Automated Benchmarks

```bash
# Incremental vs full (simulated times, fast)
poetry run python -m scripts.benchmark_incremental

# Real computational performance (actual embedding times)
poetry run python -m scripts.benchmark_compute --runs 3

# RAGAS retrieval quality
poetry run python -m scripts.benchmark_ragas

# Quality benchmarks (drift, accuracy, stability)
poetry run python -m scripts.benchmark_quality
```

### VM Testing for Reproducible Results

For the best measurement results (as would be expected in a thesis):

1. **Create a clean VM** (e.g. VirtualBox, Hyper-V, or cloud VM):
   - Ubuntu 22.04 LTS or Windows 11
   - 4+ CPU cores, 8+ GB RAM
   - No GPU needed (CPU-only embedding is fine for benchmarking)

2. **Install only the essentials**:
   ```bash
   git clone <repo-url>
   cd private-gpt
   pip install poetry
   poetry install
   poetry run pip install sentence-transformers einops psutil
   ```

3. **Run the benchmark in isolation**:
   ```bash
   # Close all other processes
   poetry run python -m scripts.benchmark_compute --runs 10
   ```

4. **Collect results**: The output CSV/JSON contains system info (CPU, RAM, OS) for reproducibility

---

## 13. File Overview

| File | Lines | Role |
|---|---|---|
| `chunk_hasher.py` | 215 | Semantic chunking + SHA-256 hashing |
| `chunk_hash_store.py` | 185 | Persistent hash registry (JSON) |
| `diff_detector.py` | 422 | Three-phase diff detection + Patience diff |
| `incremental_updater.py` | 584 | Core orchestrator (chunk -> diff -> embed -> index) |
| `incremental_watcher.py` | 218 | File watcher with debouncing |
| `ingest_component.py` | 197 | Bridge to PrivateGPT's IngestService |
| `ui.py` | — | Gradio UI with static mode indicator and per-file watcher panel |
| `settings.py` | 688 | IncrementalSettings Pydantic model |
| `benchmark_ragas.py` | 561 | RAGAS A/B evaluation script |
| `benchmark_compute.py` | 290 | Real computational performance benchmark |
| `benchmark_incremental.py` | 597 | Incremental vs full (simulated) benchmark |
| `benchmark_quality.py` | — | Quality benchmark (drift, accuracy, stability) |
| `golden_dataset.json` | 100 | 12 Q/A/Context evaluation triples |
| `test_documents/*.txt` | 4 files | Manual test documents (0%, 10%, 50%, 90% change) |

**Total**:  4,500 lines of new Python code for the complete incremental update pipeline.
