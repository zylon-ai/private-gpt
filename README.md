# PrivateGPT

<a href="https://trendshift.io/repositories/8691" target="_blank"><img src="https://trendshift.io/api/badge/repositories/8691" alt="zylon-ai%2Fprivate-gpt | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

[![Tests](https://github.com/zylon-ai/private-gpt/actions/workflows/tests.yml/badge.svg)](https://github.com/zylon-ai/private-gpt/actions/workflows/tests.yml?query=branch%3Amain)
[![Website](https://img.shields.io/website?up_message=check%20it&down_message=down&url=https%3A%2F%2Fdocs.privategpt.dev%2F&label=Documentation)](https://docs.privategpt.dev/)
[![Discord](https://img.shields.io/discord/1164200432894234644?logo=discord&label=PrivateGPT)](https://discord.gg/F8KCFeZbkx)
[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/ZylonPrivateGPT)](https://twitter.com/ZylonPrivateGPT)

**PrivateGPT is the open-source Claude API-compatible layer for building private AI applications on top of any local or self-hosted model server.**

PrivateGPT turns local inference into a real application backend. It sits above any OpenAI-compatible model server and provides the higher-level capabilities modern AI products need: messages, model selection, file ingestion, retrieval with citations, tool use, database querying, CSV and tabular analysis, web search and extraction, MCP, skills, code execution, custom tools, token counting, embeddings, and async workflows.

The goal: bring the Claude-style application API to your own infrastructure, so developers can build private AI products without depending on cloud AI APIs.

## Why PrivateGPT?

Running a model locally is only the first layer.

To build useful AI applications you need a set of high-level building blocks.
PrivateGPT provides that layer as an open-source API, so you don't need to reinvent the wheel.

- A standard messages API.
- Files and artifact ingestion.
- Retrieval with citations.
- Built-in tools, mapping those offered by Claude API.
- Custom tools support.
- MCP connectors.
- Structured access to databases and CSVs.
- Web search and extraction.
- Token counting, embeddings, and orchestration.

PrivateGPT sits between your business logic and your inference server.

```text
Your app / agent / workflow / UI
              |
        PrivateGPT API
              |
 OpenAI-compatible inference server
              |
 Local or self-hosted models
```

## PrivateGPT vs Ollama, LM Studio, LocalAI, vLLM, llama.cpp

PrivateGPT does not replace local inference providers. It complements them.

Projects like Ollama, LM Studio, LocalAI, vLLM, and llama.cpp make it possible to run and serve models locally. They answer the question:

> How do I run a model?

PrivateGPT answers the next question:

> How do I build a useful AI application on top of that model?

PrivateGPT sits above the model server and adds the application API: messages, files, retrieval, citations, tools, data analysis, MCP, skills, and custom tool execution.

In short:

```text
Ollama / LM Studio / LocalAI / vLLM / llama.cpp = local inference layer
PrivateGPT                                      = local AI application API layer
```

Use both together: run your model with the inference server you prefer, then use PrivateGPT as the Claude-style backend for your application.

## PrivateGPT vs Onyx, Open WebUI

Onyx and Open WebUI are valuable projects, but they are solving a different problem.

Onyx is an open-source AI platform focused on chat and enterprise search across organizational knowledge, documents, apps, and people. Open WebUI is a self-hosted AI interface for running and interacting with AI on your own infrastructure. Both are primarily app-first experiences.

PrivateGPT is API-first.

It is not trying to be the final workspace, enterprise search product, or ChatGPT-style interface. Instead, PrivateGPT gives developers the standardized local backend underneath those products: a Claude API-compatible layer for messages, files, retrieval, citations, tools, data analysis, MCP, skills, and custom tools.

Put differently:

```text
Onyx / Open WebUI = self-hosted AI applications
PrivateGPT       = API layer for building self-hosted AI applications
```

The lightweight UI included with PrivateGPT exists to help you test the API, demonstrate the value quickly, and explore the available capabilities. The API is the actual product.

## A Platform For Developers

PrivateGPT is built for developers who want to create private AI applications without rebuilding the same backend primitives from scratch.

With PrivateGPT, you can build:

- Local document assistants.
- Internal knowledge-base assistants.
- Claude-compatible AI features inside existing applications.
- Agents with built-in and custom tools.
- Database and CSV analysis workflows.
- Retrieval systems with citations.
- MCP-powered local AI systems.
- AI applications that run inside your own infrastructure.

PrivateGPT is intentionally backend-first. You bring your own UI, product workflow, deployment model, and inference provider.

## Test The API In The Workbench UI

PrivateGPT includes a lightweight out-of-the-box UI so you can start interacting with the API in minutes.

The UI is useful for:

- Sending messages.
- Selecting models from `/v1/models`.
- Uploading documents.
- Testing retrieval with citations.
- Enabling tools per chat.
- Configuring databases, MCP connectors, skills, and custom tools.
- Inspecting requests and responses through the API Debugger.

This UI is a demonstrator, not the core product. Developers are expected to build their own applications on top of the API. That said, the UI is intentionally polished enough for demos, videos, internal pilots, and quick local usage.

The current UI lives in:

```text
./ui/index.html
```

## PrivateGPT vs Claude API Status

PrivateGPT follows the Claude API model because Claude has become the clearest reference for modern AI application APIs: messages, tools, files, citations, skills, MCP, and structured workflows.

The goal is to support Claude API concepts as closely as possible while adapting them to local and self-hosted infrastructure. Some functionality is intentionally exposed through PrivateGPT-specific endpoints where that makes the local implementation clearer, especially around ingestion and artifact management.

Compatibility snapshot:

| Area | Capability | Claude API | PrivateGPT |
|---|---|:---:|:---:|
| Models | Model selection | ✅ | ✅ |
| Messages | Messages API | ✅ | ✅ |
| Messages | Streaming | ✅ | ✅ |
| Messages | Batch / async processing | ✅ | ✅ async |
| Messages | Token counting | ✅ | ✅ |
| Knowledge | Files / artifacts | ✅ | ✅ |
| Knowledge | PDF and document ingestion | ✅ | ✅ |
| Knowledge | Retrieval with citations | ✅ | ✅ |
| Knowledge | Embeddings | ✅ | ✅ |
| Tools | Tool use | ✅ | ✅ |
| Tools | Tools in streaming | ✅ | ✅ |
| Tools | Built-in web search | ✅ | ✅ |
| Tools | Web extraction / fetch | ✅ | ✅ |
| Tools | Custom tools | ✅ | ✅ |
| Data | Database querying | Via tools | ✅ built-in |
| Data | CSV / tabular analysis | Via tools / code | ✅ built-in |
| Agents | MCP in the API | ✅ | ✅ |
| Agents | Remote MCP servers | ✅ | ✅ |
| Agents | Skills | ✅ | ⚠️ basic |
| Output | Structured outputs | ✅ | ✅ inference-dependent|
| Models | Vision | ✅ | ✅ model-dependent |
| Optimization | Prompt caching | ✅ | ❌ |
| Reasoning | Extended thinking | ✅ | ✅ |
| Platform | Token-based auth | ✅ | ✅ |
| Platform | OAuth / organizations | ✅ | ❌ |

Legend:

- ✅ Supported.
- ⚠️ Partially supported, model-dependent, or in active development.
- ❌ Not currently supported.

The ambition is full Claude API feature coverage where it makes sense for a local, open-source application layer. Contributions are especially welcome in the areas marked as in progress.

## PrivateGPT vs Zylon.ai

<a href="https://www.zylon.ai/" target="_blank"><img src="./fern/docs/assets/zylon_banner.png" width="456" height="149"/></a>

PrivateGPT is maintained by the team behind Zylon.ai.

PrivateGPT is the open-source application API layer.
Zylon is the complete, enterprise-ready on-prem AI platform built around it.

PrivateGPT gives developers the Claude API-compatible backend primitives: messages, ingestion, tools, retrieval, citations, database access, tabular analysis, MCP, skills, and custom tools.

Zylon turns that layer into a complete production platform for regulated organizations. It adds the pieces enterprises need to deploy and operate private AI at scale:

- Integrated inference server.
- Kubernetes deployment package.
- API gateway and developer platform.
- Workspace application for non-technical end users.
- CLI for installation, updates, model selection, and platform configuration.
- LDAP integration for user management.
- Telemetry and operational monitoring.
- SIEM audit logs for compliance.
- SharePoint, Confluence, FTP, and Samba document source integrations.
- Disconnected operation without external cloud dependencies.
- More than 20 production services packaged together.
- Integrated n8n Community Edition for workflow automation.

Use PrivateGPT if you want the open-source local AI application layer and developer API.

Use Zylon if you need the full enterprise platform around it: deployment, governance, operations, user management, integrations, auditability, and support.

Learn more at [zylon.ai](https://zylon.ai) or [book a demo](https://cal.com/zylon/demo?source=privategptreadme).
