# PrivateGPT Examples

## AG2 Multi-Agent Chat

[`ag2_multiagent.py`](ag2_multiagent.py) demonstrates how to use [AG2](https://ag2.ai) multi-agent orchestration on top of PrivateGPT. Multiple specialized AI agents collaborate to analyze your private documents — all data stays local.

### How it works

```
User question → AG2 GroupChat → PrivateGPT API (localhost)
  ├── Researcher: RAG queries over your documents
  ├── Analyst: follow-up queries to verify and expand
  └── Writer: synthesizes findings into a final answer
```

AG2 agents use PrivateGPT's OpenAI-compatible `/v1/chat/completions` endpoint as a tool. The agents reason using an external LLM (e.g., GPT-4o-mini) but retrieve all document context from your local PrivateGPT instance.

### Setup

1. **Start PrivateGPT** with an OpenAI-compatible LLM:
   ```bash
   PGPT_PROFILES=openai make run
   ```

2. **Ingest documents** via the PrivateGPT UI (http://localhost:8001) or API.

3. **Install AG2**:
   ```bash
   pip install "ag2[openai]>=0.11.4,<1.0"
   ```

4. **Run the example**:
   ```bash
   export OPENAI_API_KEY="your-key"
   export QUERY="What are the main topics covered in the documents?"
   python examples/ag2_multiagent.py
   ```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | API key for the agent reasoning LLM |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for agent reasoning |
| `PRIVATEGPT_URL` | `http://localhost:8001/v1` | PrivateGPT server URL |
| `PRIVATEGPT_API_KEY` | `not-required` | PrivateGPT API key (if auth enabled) |
| `QUERY` | Summary prompt | The question to ask |

### Privacy model

- **Document retrieval**: 100% local via PrivateGPT (no data leaves your machine)
- **Agent reasoning**: Uses external LLM (OpenAI by default). For fully private operation, point `OPENAI_MODEL` to a local model via Ollama or similar.
