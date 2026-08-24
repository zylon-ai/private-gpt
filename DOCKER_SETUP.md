# PrivateGPT Docker-Only Setup

This guide runs both the model server and PrivateGPT in Docker. No host Python or
`uv` installation is required.

## Recommended architecture

```text
Browser
  |
  | http://127.0.0.1:8080
  v
PrivateGPT container
  |
  | Docker network
  v
Ollama container
  |-- qwen3.5:35b
  `-- mxbai-embed-large
```

Ollama is the simplest inference provider for an initial local deployment. For a
multi-user production deployment or reliable JSON-schema output, consider vLLM
instead.

## Requirements

- Docker Engine with the Compose plugin
- NVIDIA GPU driver
- NVIDIA Container Toolkit configured for Docker
- Approximately 30 GB of free disk space for images, models, and working data
- Internet access during the initial image and model downloads

No host Python environment is needed.

Verify Docker and GPU access before continuing:

```bash
docker version
docker compose version
docker run --rm --gpus all \
  nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

## Compose configuration

Create `compose.yaml` in the directory from which you want to operate the
deployment:

```yaml
name: privategpt

services:
  ollama:
    image: ollama/ollama:latest
    container_name: privategpt-ollama
    restart: unless-stopped
    gpus: all
    volumes:
      - ollama-data:/root/.ollama
    networks:
      - privategpt

  private-gpt:
    image: zylonai/private-gpt:latest
    container_name: privategpt-api
    restart: unless-stopped
    depends_on:
      - ollama
    ports:
      - "127.0.0.1:8080:8080"
    environment:
      OPENAI_API_BASE: http://ollama:11434/v1
      OPENAI_EMBEDDING_API_BASE: http://ollama:11434/v1
      PGPT_LLM_DEFAULT: qwen3.5:35b
      PGPT_EMBEDDING_DEFAULT: mxbai-embed-large
    volumes:
      - privategpt-data:/home/worker/app/local_data
    networks:
      - privategpt

networks:
  privategpt:
    driver: bridge

volumes:
  ollama-data:
  privategpt-data:
```

The Ollama API is deliberately not published on a host port. PrivateGPT reaches
it through the dedicated Docker network, while Ollama retains outbound access
for model downloads. PrivateGPT is bound to `127.0.0.1`, so it is available only
on the local computer.

For repeatable deployments, replace both `latest` tags with tested, pinned image
versions.

## Start Ollama and download the models

Start only the inference service first:

```bash
docker compose up -d ollama
```

Download the recommended models:

```bash
docker compose exec ollama ollama pull qwen3.5:35b
docker compose exec ollama ollama pull mxbai-embed-large
```

The example LLM download is approximately 24 GB. The embedding model is
approximately 670 MB.

Confirm that both are installed:

```bash
docker compose exec ollama ollama list
```

## Start PrivateGPT

```bash
docker compose up -d private-gpt
docker compose ps
```

Follow startup logs if necessary:

```bash
docker compose logs -f private-gpt
```

Open the Workbench UI:

```text
http://127.0.0.1:8080/ui
```

The API is available at:

```text
http://127.0.0.1:8080
```

## Verify the deployment

Check the PrivateGPT health endpoint:

```bash
curl --fail http://127.0.0.1:8080/health
```

Check model discovery through the internal Ollama endpoint:

```bash
docker compose exec private-gpt \
  python -c "import requests; print(requests.get('http://ollama:11434/v1/models', timeout=30).json())"
```

## Operations

Stop the containers without deleting their data:

```bash
docker compose down
```

Start them again:

```bash
docker compose up -d
```

Review resource use:

```bash
docker stats privategpt-ollama privategpt-api
nvidia-smi
```

Pull updated images and recreate the containers:

```bash
docker compose pull
docker compose up -d
```

## Persistence and backup

The named volumes preserve:

- `ollama-data`: downloaded model files
- `privategpt-data`: ingested documents, Qdrant data, and application state

`docker compose down` preserves these volumes. Commands such as
`docker compose down --volumes` delete them and should only be used when their
data is no longer needed.

Back up `privategpt-data` before upgrades or major configuration changes.

## Security recommendations

- Keep the `127.0.0.1:8080:8080` binding for personal use.
- Do not publish Ollama port `11434` unless another trusted host must reach it.
- If PrivateGPT must be reachable over a LAN or the internet, put it behind an
  authenticated TLS reverse proxy.
- Enable PrivateGPT authentication and replace the example/default secret.
- Restrict CORS to the exact client origins instead of allowing `*`.
- Do not expose local code-execution, web, database, or MCP tools to untrusted
  users without appropriate isolation and access controls.

## Provider choice

| Provider | Recommended use | Main tradeoff |
|---|---|---|
| Ollama | Initial setup and single-user local use | Approximate token counting and no enforced JSON-schema output |
| vLLM | Production, concurrency, and structured tool calls | More GPU configuration and operational complexity |
| llama.cpp | CPU/GPU flexibility and GGUF models | Lower multi-user throughput than vLLM |

With a 32 GB RTX 5090, `qwen3.5:35b` is a suitable starting model. Keep the
context window and concurrency conservative if GPU memory becomes constrained.
