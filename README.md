# PrivateGPT

[![Tests](https://github.com/zylon-ai/private-gpt/actions/workflows/tests.yml/badge.svg)](https://github.com/zylon-ai/private-gpt/actions/workflows/tests.yml?query=branch%3Amain)
[![Discord](https://img.shields.io/discord/1164200432894234644?logo=discord&label=PrivateGPT)](https://discord.gg/bK6mRVpErU)

**PrivateGPT** is a production-ready, 100% private AI system for querying your documents with Large Language Models. No data ever leaves your machine.

Built on **FastAPI** + **LlamaIndex** with a **Chainlit** web UI and a fully **OpenAI-compatible REST API**.

---

## System Components

Understanding the components is the key to configuring PrivateGPT for your use case. Every component is swappable via `settings.yaml`.

### 1. LLM Component

The language model that generates responses. Configured under `llm.mode`.

| Mode | Provider | Use Case |
|------|----------|----------|
| `llamacpp` | Local GGUF model via llama.cpp | Fully offline, no API key |
| `ollama` | Local models via Ollama | Easy local model management |
| `openai` | OpenAI API (GPT-4, etc.) | Cloud, best quality |
| `azopenai` | Azure OpenAI | Enterprise cloud |
| `openailike` | Any OpenAI-compatible API | LM Studio, vLLM, etc. |
| `gemini` | Google Gemini API | Cloud alternative |
| `sagemaker` | AWS SageMaker endpoint | AWS deployments |

### 2. Embedding Component

Converts text to vectors for semantic search. Configured under `embedding.mode`.

| Mode | Provider | Notes |
|------|----------|-------|
| `huggingface` | Local HuggingFace model | Default, fully offline |
| `ollama` | Via Ollama | Uses same Ollama instance |
| `openai` | OpenAI embeddings | `text-embedding-ada-002` |
| `azopenai` | Azure OpenAI | Enterprise |
| `gemini` | Google Gemini | Cloud |
| `mistralai` | Mistral AI | Cloud |
| `sagemaker` | AWS SageMaker | AWS |

### 3. Vector Store

Stores and searches document embeddings. Configured under `vectorstore.database`.

| Database | Type | Notes |
|----------|------|-------|
| `postgres` | PostgreSQL + pgvector | **Default** — persistent, SQL-queryable |
| `qdrant` | Qdrant | High-performance vector DB |
| `chroma` | ChromaDB | Embedded, simple |
| `milvus` | Milvus | Scalable, local SQLite |
| `clickhouse` | ClickHouse | Analytics-optimized |

### 4. Node / Document Store

Stores document chunks and metadata. Configured under `nodestore.database`.

| Database | Type | Notes |
|----------|------|-------|
| `postgres` | PostgreSQL | **Default** — persistent, SQL-queryable |
| `simple` | Local filesystem | No external DB needed |

### 5. RAG Pipeline

The Retrieval Augmented Generation pipeline, built on LlamaIndex:

1. **Ingest** — Document is parsed, split into chunks, embedded, and stored
2. **Retrieve** — Query is embedded, top-k similar chunks are fetched from vector store
3. **Augment** — Retrieved chunks are injected into the LLM prompt as context
4. **Generate** — LLM produces a grounded answer with source citations

Configured under `rag.similarity_top_k` and `rag.rerank`.

### 6. REST API

Full OpenAI-compatible API on port `8001`. All endpoints support streaming.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat with optional RAG context |
| `/v1/completions` | POST | Text completion |
| `/v1/embeddings` | POST | Generate embeddings |
| `/v1/ingest/file` | POST | Upload and ingest a document |
| `/v1/ingest/list` | GET | List all ingested documents |
| `/v1/ingest/{doc_id}` | DELETE | Delete an ingested document |
| `/v1/chunks/retrieve` | POST | Retrieve relevant chunks |
| `/v1/recipes/summarize` | POST | Summarize documents |
| `/v1/auth/*` | POST | User/group/collection management |
| `/health` | GET | Health check |

Interactive API docs: `http://localhost:8001/docs`

### 7. Web UI (Chainlit)

A Chainlit-based web interface mounted at the configured `ui.path` (default `/`).

**Chat Modes:**
- **RAG** — Answer questions using your documents as context; sources are cited
- **Search** — Find the most relevant passages across all ingested documents
- **Basic** — Chat directly with the AI without any document context
- **Summarize** — Generate a comprehensive summary of a selected file

**Features:**
- File upload via the 📎 attachment button
- File management (list, select, delete) via action buttons
- Configurable system prompt per mode
- Optional user authentication with per-collection access control
- Admin panel for managing users, groups, and collections

---

## Quick Start

### Prerequisites

- Python 3.11
- [Poetry](https://python-poetry.org/) 1.8+
- PostgreSQL 16 with pgvector (or use Docker Compose)
- Ollama (for local LLMs) — or an OpenAI API key

### 1. Install Dependencies

```bash
# Core + UI + PostgreSQL stores + Ollama LLM/embeddings
poetry install --extras "ui llms-ollama embeddings-ollama vector-stores-postgres storage-nodestore-postgres"

# For local models (llama.cpp):
poetry install --extras "ui llms-llama-cpp embeddings-huggingface vector-stores-postgres storage-nodestore-postgres"

# For OpenAI:
poetry install --extras "ui llms-openai embeddings-openai vector-stores-postgres storage-nodestore-postgres"
```

### 2. Start PostgreSQL

Using Docker Compose (recommended):

```bash
docker-compose up postgres -d
```

Or connect to an existing PostgreSQL instance by setting environment variables:

```bash
export POSTGRES_HOST=your-host
export POSTGRES_USER=your-user
export POSTGRES_PASSWORD=your-password
export POSTGRES_DB=your-database
```

> **HeiSQL / pgAdmin / DBeaver:** Connect to `localhost:5432` with the credentials above. The schema `private_gpt` will be created automatically on first run.

### 3. Configure

Edit `settings.yaml` or set environment variables. The minimum configuration for Ollama:

```yaml
llm:
  mode: ollama

embedding:
  mode: ollama

ollama:
  llm_model: llama3.1
  embedding_model: nomic-embed-text
  api_base: http://localhost:11434
```

For OpenAI, set `OPENAI_API_KEY` and use `settings-openai.yaml`:

```bash
PGPT_PROFILES=openai python -m private_gpt
```

### 4. Run

```bash
python -m private_gpt
```

Open `http://localhost:8001` in your browser to access the Chainlit UI.

### 5. Docker Compose (Full Stack)

```bash
# Ollama + PostgreSQL (default)
docker-compose up

# With NVIDIA GPU
docker-compose --profile ollama-cuda up
```

---

## How to Use

### Uploading Documents

**Via UI:** Click the 📎 attachment button in the Chainlit chat interface and attach your files. Supported formats: PDF, DOCX, TXT, HTML, Markdown, and more.

**Via API:**
```bash
curl -X POST http://localhost:8001/v1/ingest/file \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-document.pdf"
```

**Bulk ingest from a folder:**
```bash
python scripts/ingest_folder.py /path/to/your/documents
```

### Chat Modes

Switch modes using the ⚙ Settings panel in the UI.

**RAG Mode** — Query your documents:
```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What does the report say about Q3 revenue?"}],
    "use_context": true,
    "stream": false
  }'
```

**Basic Mode** — Chat without documents:
```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain what a RAG pipeline is."}],
    "use_context": false,
    "stream": false
  }'
```

**Streaming:**
```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Summarize the key findings."}],
    "use_context": true,
    "stream": true
  }'
```

**Search / Chunk Retrieval:**
```bash
curl -X POST http://localhost:8001/v1/chunks/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "text": "revenue growth",
    "limit": 5
  }'
```

**Summarize:**
```bash
curl -X POST http://localhost:8001/v1/recipes/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "use_context": true,
    "context_filter": {"docs_ids": ["your-doc-id"]}
  }'
```

### Managing Ingested Documents

```bash
# List all documents
curl http://localhost:8001/v1/ingest/list

# Delete a document
curl -X DELETE http://localhost:8001/v1/ingest/{doc_id}
```

---

## Configuration Reference

### Key settings.yaml Sections

```yaml
server:
  port: 8001                    # API port

llm:
  mode: ollama                  # LLM provider
  max_new_tokens: 512           # Max tokens per response
  temperature: 0.1              # Creativity (0=deterministic, 1=creative)

embedding:
  mode: ollama                  # Embedding provider
  embed_dim: 768                # Must match your embedding model

rag:
  similarity_top_k: 2          # Number of chunks retrieved per query
  rerank:
    enabled: false              # Enable reranker for better precision

vectorstore:
  database: postgres            # Vector store backend

nodestore:
  database: postgres            # Document store backend

postgres:
  host: ${POSTGRES_HOST:localhost}
  port: ${POSTGRES_PORT:5432}
  database: ${POSTGRES_DB:postgres}
  user: ${POSTGRES_USER:postgres}
  password: ${POSTGRES_PASSWORD:postgres}
  schema_name: private_gpt      # PostgreSQL schema

ui:
  enabled: true
  path: /                       # URL path for the Chainlit UI
  default_mode: "RAG"           # Starting mode
```

### Environment Variables

All `settings.yaml` values support `${VAR:default}` syntax for environment variable overrides:

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | API server port | `8001` |
| `POSTGRES_HOST` | PostgreSQL host | `localhost` |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `POSTGRES_DB` | Database name | `postgres` |
| `POSTGRES_USER` | Database user | `postgres` |
| `POSTGRES_PASSWORD` | Database password | `postgres` |
| `OPENAI_API_KEY` | OpenAI API key | *(empty)* |
| `HF_TOKEN` | HuggingFace token (for gated models) | *(empty)* |
| `ADMIN_PASSWORD` | Default admin password | `changeme` |
| `PGPT_PROFILES` | Additional settings profiles | *(empty)* |

### Profiles

Load additional settings files via `PGPT_PROFILES`:

```bash
# Use settings-openai.yaml on top of settings.yaml
PGPT_PROFILES=openai python -m private_gpt

# Multiple profiles (comma-separated)
PGPT_PROFILES=openai,docker python -m private_gpt
```

Preset profiles available: `ollama`, `local`, `openai`, `azopenai`, `gemini`, `docker`.

---

## Database Setup (PostgreSQL)

PrivateGPT uses PostgreSQL for both the **vector store** (embeddings) and the **document store** (metadata, chunks).

### Requirements

PostgreSQL 14+ with the **pgvector** extension. The easiest way is to use the official Docker image:

```bash
docker run -d \
  --name privategpt-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### Schema

On first run, PrivateGPT automatically creates the `private_gpt` schema and all required tables:

- `private_gpt.data_embeddings` — vector embeddings (pgvector)
- `private_gpt.data_docstore` — document chunk content
- `private_gpt.data_indexstore` — index metadata

### Connecting with HeidiSQL / DBeaver / pgAdmin

| Setting | Value |
|---------|-------|
| Host | `localhost` |
| Port | `5432` |
| Database | `postgres` |
| User | `postgres` |
| Password | `postgres` |
| Schema | `private_gpt` |

You can query, inspect, and edit data directly with any PostgreSQL client.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PrivateGPT                           │
│                                                         │
│  ┌──────────┐   ┌────────────────────────────────────┐ │
│  │ Chainlit │   │         FastAPI REST API            │ │
│  │   UI     │──▶│  /v1/chat  /v1/ingest  /v1/chunks  │ │
│  └──────────┘   └────────────────┬───────────────────┘ │
│                                  │                      │
│                    ┌─────────────▼─────────────┐        │
│                    │      LlamaIndex RAG        │        │
│                    │  Ingest → Embed → Retrieve │        │
│                    └──────┬──────────┬──────────┘        │
│                           │          │                   │
│              ┌────────────▼──┐  ┌────▼────────────────┐ │
│              │  LLM Component│  │ Embedding Component  │ │
│              │ (Ollama/OpenAI│  │ (Ollama/HuggingFace/ │ │
│              │  /LlamaCPP/…) │  │  OpenAI/…)           │ │
│              └───────────────┘  └──────────────────────┘ │
│                                                         │
│  ┌──────────────────────┐  ┌──────────────────────────┐ │
│  │   Vector Store       │  │   Document Store         │ │
│  │ PostgreSQL (pgvector)│  │   PostgreSQL             │ │
│  └──────────────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Code Structure

```
private_gpt/
├── __main__.py              # Entry point (uvicorn)
├── main.py                  # FastAPI app
├── launcher.py              # App factory + router registration
├── di.py                    # Dependency injection wiring
├── settings/
│   └── settings.py          # Pydantic config models
├── components/              # Pluggable infrastructure
│   ├── llm/                 # LLM providers
│   ├── embedding/           # Embedding providers
│   ├── vector_store/        # Vector store backends
│   ├── node_store/          # Document store backends
│   └── ingest/              # Ingestion pipeline
├── server/                  # FastAPI routers + services
│   ├── chat/                # Chat & completions
│   ├── ingest/              # Document ingestion
│   ├── chunks/              # Chunk retrieval
│   ├── embeddings/          # Embeddings API
│   ├── recipes/summarize/   # Summarization
│   └── auth/                # User/group/collection auth
└── ui/
    ├── ui.py                # PrivateGptUi (Chainlit mount)
    └── chainlit_app.py      # Chainlit event handlers
```

---

## Contributing

Contributions are welcome! Run checks before submitting:

```bash
make check    # Format + type check
make test     # Run tests
```

See the [Project Board](https://github.com/users/imartinez/projects/3) for open issues.

## Community

- [Discord](https://discord.gg/bK6mRVpErU)
- [Twitter / X](https://twitter.com/ZylonPrivateGPT)

## Citation

```bibtex
@software{Zylon_PrivateGPT_2023,
  author = {Zylon by PrivateGPT},
  license = {Apache-2.0},
  month = may,
  title = {{PrivateGPT}},
  url = {https://github.com/zylon-ai/private-gpt},
  year = {2023}
}
```

## Partners

- [Qdrant](https://qdrant.tech/) — Vector database
- [LlamaIndex](https://www.llamaindex.ai/) — RAG framework
- [Fern](https://buildwithfern.com/) — Documentation & SDKs
