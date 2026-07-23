# PrivateGPT Workbench PRD

## Project Context

PrivateGPT is an open-source local AI API project. Its value proposition is not local inference itself, but the higher-level application layer built on top of any OpenAI-compatible local inference backend.

PrivateGPT aims to provide a local implementation of the capabilities developers and users expect from modern Claude-style APIs and applications, including:

- Chat/messages API
- File ingestion
- Retrieval with citations
- Text-to-SQL / database querying
- CSV and tabular analysis through sandboxed Python execution
- Web search and web extraction
- MCP support
- Skills
- Custom tools
- Embeddings and lower-level primitives

The demonstrator UI, tentatively called **PrivateGPT Workbench**, exists to make these API capabilities tangible.

For non-technical users, it should show PrivateGPT as:

> A free, local AI assistant I can use to query documents, knowledge bases, websites, CSVs, and databases without relying on a cloud API key.

For developers, it should show PrivateGPT as:

> A local Claude-compatible API layer I can build applications on top of.

This UI should remain a lightweight demonstrator, not the main product. It now lives inside the PrivateGPT repository under `./ui` and must not become a heavy frontend application or a maintenance burden.

## OpenAPI Source Of Truth

The API contract is defined by the Fern-generated repository-root relative OpenAPI file:

```text
./fern/openapi/openapi.json
```

From the `./ui` directory, the same file resolves as:

```text
../fern/openapi/openapi.json
```

The Fern-generated OpenAPI file is the primary source of truth for:

- Available endpoints
- Request body shapes
- Response body shapes
- Schema names
- Supported tool/context structures
- Message block formats
- Artifact formats
- MCP/tool fields
- Future API changes

The implementation must inspect and follow the Fern-generated OpenAPI schema rather than relying only on this PRD's examples. Examples in this PRD are illustrative and should be corrected wherever the API contract differs.

Important current endpoints include:

```text
POST /v1/messages
POST /v1/messages/count_tokens
POST /v1/messages/validate
GET  /v1/models
POST /v1/artifacts/ingest
GET  /v1/artifacts/list?collection=<collection>
POST /v1/artifacts/delete
POST /v1/artifacts/content
POST /v1/artifacts/chunked-content
POST /v1/primitives/search
POST /v1/tools/semantic-search
POST /v1/tools/tabular-data-analysis
POST /v1/tools/database-query
POST /v1/tools/web-fetch
POST /v1/tools/web-search
```

The implementation should treat `POST /v1/messages` as the central endpoint. Most of the product experience should flow through chat, with Context providing the inputs and Debugger explaining the underlying API interactions.

Before implementing request builders, parse or manually inspect `../fern/openapi/openapi.json` and align all payloads with the current schemas, especially:

- `ChatBody`
- `MessageInput`
- `ToolSpecBody`
- `ContextFilter`
- `FileArtifact`
- `SqlDatabaseArtifact`
- `McpServerConfig`
- Tool response block schemas

## Visual And UX Source Of Truth

The product and implementation requirements in this PRD should be paired with the visual direction in this repository-root relative file:

```text
./ui/docs/STYLE_GUIDE.md
```

From the `./ui` directory, that file resolves as:

```text
./docs/STYLE_GUIDE.md
```

That style guide includes repo-local reference images copied from the current brand/UI explorations:

```text
./ui/references/primary-chat-layout.png
./ui/references/search-overlay.png
./ui/references/chat-tools-composer.png
./ui/references/context-knowledge-base.png
```

From `./ui`, those files resolve as:

```text
../references/primary-chat-layout.png
../references/search-overlay.png
../references/chat-tools-composer.png
../references/context-knowledge-base.png
```

Use the style guide as the source of truth for layout, glass surfaces, background treatment, sidebar behavior, chat composer treatment, Context rows, and Debugger visual density.

## Purpose

PrivateGPT Workbench is a lightweight demonstrator UI for the PrivateGPT API. It should prove PrivateGPT's value as a local Claude-compatible AI application backend while staying simple enough to live inside the PrivateGPT repo as a non-core demo.

The app is not intended to become a full product, admin console, or design-system-heavy frontend. It is a practical local UI for trying the API, showing non-technical users what PrivateGPT can do, and helping developers understand how to build on top of it.

## Primary Goals

1. Let non-technical users experience PrivateGPT as a local AI assistant.
2. Let users configure the local context available to the assistant: documents, databases, web search, MCP, skills, and custom tools.
3. Let developers observe how the UI talks to the API through a lightweight session-level API Debugger.
4. Keep implementation simple: ideally a single static HTML file with vanilla JS/CSS and browser `localStorage`.

## Non-Goals

- No user accounts.
- No server-side UI database.
- No projects, folders, organizations, or workspace hierarchy.
- No complex design system.
- No cloud sync.
- No heavy frontend framework unless absolutely necessary.
- No attempt to replace browser DevTools.
- No permanent debugger storage.

## Architecture

Recommended initial implementation:

```text
ui/
  index.html
```

Single-file static app containing:

- HTML
- CSS
- JavaScript

Browser storage:

- `localStorage` for persistent app state.
- In-memory state for debugger events.

Default PrivateGPT API base URL:

```text
http://127.0.0.1:8001
```

Allow user override from inside the web application. Users must not need to edit config files to point the demonstrator at a different PrivateGPT deployment.

Connection settings should include:

- PrivateGPT API base URL
- Optional PrivateGPT API key / bearer token

If auth is configured, requests should include:

```text
Authorization: Basic <base64(username:password)>
```

### PrivateGPT API vs LLM Gateway

There are two separate connection concepts:

1. **PrivateGPT API URL and auth**
   - This is what Workbench calls directly.
   - This must be configurable in the Workbench UI.
   - Default URL: `http://127.0.0.1:8001`.
   - Supported auth in the Workbench UI: optional HTTP Basic auth only, entered as username and password fields.

2. **LLM Gateway URL and auth**
   - This points PrivateGPT to the underlying inference provider, such as a local Ollama server.
   - This is configured in PrivateGPT's own configuration files.
   - Workbench must not expose LLM gateway configuration in the UI for v1.
   - A common local LLM gateway default may be Ollama at `http://127.0.0.1:11434`, but that belongs to PrivateGPT backend configuration, not Workbench.

All Workbench API calls should target the configured PrivateGPT API base URL. Do not call the LLM gateway directly from the browser UI.

For development and automated testing, a working PrivateGPT deployment may be available through local environment variables:

```text
PGPT_BASE_URL
PGPT_TOKEN
```

The implementation may read these at runtime in local test scripts or dev-server setup, but must never store, print, commit, or hardcode the actual values.

Browser/CORS behavior should prioritize the local case first. The app should work when served locally against a local PrivateGPT API. If deployed elsewhere, it should still allow the user to configure API URL and token, but any required cross-origin server policy must be handled by the PrivateGPT deployment.

## High-Level UI

The app has a persistent left sidebar and a main content area.

```text
Sidebar
  Context
  New Chat

  Chats
    Contract review
    CSV analysis
    Database demo
    Custom tool test

  API Debugger
  Settings
  GitHub
  Not for Production

Main
  If first launch or onboarding restarted:
    Guided onboarding overlay
      Step 1: URL + collection + live checks
      Step 2: Optional look-and-feel customization

  If Context selected:
    Context configuration screen

  If Settings selected:
    PrivateGPT API connection settings and assistant behavior

  If API Debugger selected:
    Session-level API request/response trace

  If Chat selected:
    Chat interface
```

## Persistent State

Store in `localStorage`:

```ts
{
  privateGptBaseUrl: string,
  privateGptUsername: string,
  privateGptPassword: string,
  systemPrompt: string,
  useCitations: boolean,
  selectedModel: string | null,
  uiAppearance: {
    brief: string,
    brandName: string,
    welcomeTitle: string,
    welcomeSubtitle: string,
    customInstructions: string,
    palette: {
      accent: string,
      secondary: string,
      surface: string,
      background: string
    },
    features: {
      databases: boolean,
      web: boolean,
      mcp: boolean,
      skills: boolean,
      customTools: boolean,
      apiDebugger: boolean,
      github: boolean,
      productionNotice: boolean
    }
  },
  onboarding: {
    completed: boolean,
    step: 1 | 2,
    appearanceSkipped: boolean,
    lastCheck: {
      ok: boolean,
      testedAt: string,
      summary: string,
      steps: Array<{ label: string, detail: string, ok: boolean | null }>
    } | null
  },

  context: {
    documents: {
      defaultCollection: string
    },
    databases: DatabaseConfig[],
    mcpServers: McpServerConfig[],
    skills: SkillConfig[],
    customTools: CustomToolConfig[]
  },

  chats: ChatSession[],
  activeChatId: string | null
}
```

Chat session:

```ts
type ChatSession = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;

  messages: ChatMessage[];

  settings: {
    enabledDocuments: boolean;
    enabledDatabases: string[];
    enabledWeb: boolean;
    enabledMcpServers: string[];
    enabledSkills: string[];
    enabledCustomTools: string[];
    model: string | null;
  };
};
```

Do not store debugger data.

On reload:

- Onboarding stays dismissed after successful completion.
- Context comes back.
- Chat list comes back.
- Chat messages come back.
- Chat-specific tool toggles come back.
- Appearance overrides and optional feature visibility come back.
- Debugger is empty.

## Sidebar Behavior

Sidebar items:

1. **Context**
   - Opens global Context screen.

2. **New Chat**
   - Creates new chat session.
   - Sets default title like `New chat`.
   - Copies default context settings from global defaults.
   - Opens the new chat.

3. **Chat List**
   - Shows persisted chats sorted by `updatedAt DESC`.
   - Selecting a chat opens it.
   - Current chat is highlighted.
   - Allow rename/delete through small inline menu or simple buttons.
   - Long chat titles must truncate with ellipsis.
   - The sidebar and chat list must never show a horizontal scrollbar.

4. **API Debugger**
   - Opens the session-level request/response trace.
   - Place this in the bottom sidebar group above Settings.

5. **Settings**
   - Opens PrivateGPT API connection settings.
   - Place this in the bottom sidebar group below API Debugger.

6. **GitHub**
   - Links to the PrivateGPT repository.
   - Shows live repository stars when GitHub is reachable.
   - Place this below Settings.

7. **Not for Production**
   - Opens a modal explaining that this UI is a local demonstrator, not a production-ready application.
   - Place this below the GitHub widget.

No projects or grouping.

## Settings Screen

The Settings screen owns Workbench-level configuration that is not part of assistant context.

Settings include:

- PrivateGPT API base URL.
- Optional HTTP Basic auth (username and password).
- Optional system prompt.
- Optional workspace instructions prepended ahead of the system prompt.
- Use citations toggle, enabled by default.
- **Collection** — the active document collection name used for document ingestion, listing, deletion, search, and chat requests. This field belongs in Settings because it is a global Workbench-level pointer, not a per-context source configuration.
- Look-and-feel overrides for brand copy, welcome copy, palette, and optional visible sections.
- Run onboarding again.
- Test API connection.
- Save settings.
- Clear local browser data.

The PrivateGPT API settings should not live in the Context screen. Context is for sources/tools the assistant can use; Settings is for how Workbench connects to PrivateGPT.

The system prompt also belongs in Settings because it controls global assistant behavior. Send it through the top-level `system` field in `ChatBody`, not as a `system` role message. If the user leaves it empty, do not send system prompt text. A `system` object may still be sent without `text` when needed for request-level options such as `citations.enabled`.

The Use citations toggle controls whether document-enabled chats request citation-annotated answers. It should be enabled by default.

The Clear local data action removes this Workbench's saved chats, settings, token, context, and preferences from browser `localStorage`. It must not imply deletion of data stored in PrivateGPT itself, such as ingested documents or backend configuration.

## Onboarding

On first launch, Workbench should open an onboarding overlay before normal use.

Step 1 requirements:

- Collect PrivateGPT base URL.
- Collect optional HTTP Basic auth.
- Collect the active collection ID.
- Save those values into the same persistent state used by Settings.
- Run live checks against:
  - `GET /v1/models`
  - `GET /v1/artifacts/list?collection=<collection>`
  - `GET /v1/skills?collection=<collection>`
- Show a simple pass/fail checklist and block progression until the checks pass.

Step 2 requirements:

- Must be skippable.
- Must accept a natural-language theme brief and use the configured LLM through `POST /v1/messages` to generate a starting appearance proposal.
- Must show the generated result before the user finishes onboarding.
- Must allow customizing brand name, welcome text, workspace instructions, colors, and optional visible sections after generation.
- GitHub/Zylon references must remain visible and must not be removable through generated or manual appearance settings.
- Must write into persistent appearance variables that override the runtime UI state and CSS custom properties.
- Must remain editable later from Settings or by rerunning onboarding.

## Not For Production Disclosure

The sidebar should include a persistent **Not for Production** disclosure below the GitHub widget.

Clicking it opens a closable glass-style modal titled:

```text
This demonstrator is not intended for Production use
```

The modal should explain that Workbench is useful for trying API capabilities, debugging requests, and exploring local AI workflows, but should not be published as a production application.

The disclosure should cover four concise risks:

- Browser storage is not secure secret storage. The bearer token, chats, context configuration, and settings are saved in `localStorage`.
- There is no application-level access control. Anyone with access to the page can use the configured API endpoint, token, tools, documents, and model access.
- Debugging data is intentionally visible. API Debugger can show prompts, document excerpts, headers, metadata, requests, and responses.
- Custom tools run browser JavaScript. Only trusted code should be used, and the UI should not be exposed without a reviewed deployment model.

End the modal with a commercial Zylon CTA:

- Link `Zylon` to `https://zylon.ai`.
- Link `book a demo with our team` to `https://cal.com/zylon/demo?source=privategptui`.
- Use this text: `Zylon is an enterprise AI platform delivering on-premise generative AI infrastructure for regulated industries, enabling secure deployment without external cloud dependencies.`

## Context Screen

The Context screen defines what the assistant can use.

Sections:

```text
Documents
Databases
Web
MCP
Skills
Custom Tools
```

A compact tab or accordion layout is acceptable.

### Documents

Purpose: manage ingested local knowledge.

Capabilities:

- Set collection name.
- Upload local files.
- Ingest files through `POST /v1/artifacts/ingest`.
- List ingested artifacts through `GET /v1/artifacts/list?collection=<collection>`.
- Delete artifacts through `POST /v1/artifacts/delete`.
- Optional diagnostic search box using `POST /v1/tools/semantic-search`.
- Optional content preview using `POST /v1/artifacts/content`.

The collection field is important and must be user-configurable. Some PrivateGPT deployments create collections on the fly, while others sit behind a gateway that restricts each bearer token to one or more allowed collections. If a deployment enforces allowed collections, ingest/list/search calls must use one of those allowed collection ids or the API may reject the request.

The Collection field lives in **Settings**, not in the Documents panel. It applies globally and is used for:

- Document ingestion.
- Document listing.
- Document deletion.
- Document search.
- Document-enabled chat requests.

There is one active collection for the Workbench. Do not expose a separate chat-level collection selector in v1.

Upload behavior:

- Convert file to base64.
- Use an artifact id derived from filename plus timestamp or UUID.
- Include metadata with `file_name`.

Example ingest body:

```json
{
  "artifact": "contract-2026-05-14",
  "collection": "default",
  "input": {
    "type": "file",
    "value": "<base64>"
  },
  "metadata": {
    "file_name": "contract.pdf"
  }
}
```

### Documents In Chat

When a chat has Documents enabled, the `/v1/messages` call should enable the semantic search tool and scope it to the configured Documents collection.

Use this request pattern:

```json
{
  "model": "default",
  "messages": [
    {
      "role": "user",
      "content": "Find the property address in the documents. Answer just with the address, no extra text."
    }
  ],
  "tools": [
    {
      "name": "semantic_search",
      "type": "semantic_search_v1"
    }
  ],
  "tool_context": [
    {
      "type": "ingested_artifact",
      "context_filter": {
        "collection": "<configured-collection>",
        "artifacts": []
      }
    }
  ]
}
```

An empty `artifacts` array means search all documents within the configured collection.

### Databases

Purpose: define SQL database artifacts available to chat.

Stored locally only.

Fields:

- `id`
- `name`
- `connection_string`
- `description`
- `schemas`, optional comma-separated list
- `ssl`
- `enable_tables`
- `enable_views`
- `enable_functions`
- `enable_procedures`

When selected in chat, convert database configs into `tool_context` artifacts:

```json
{
  "type": "sql_database",
  "connection_string": "...",
  "schemas": null,
  "ssl": false,
  "enable_tables": true,
  "enable_views": true,
  "enable_functions": true,
  "enable_procedures": true,
  "description": "Local sales database"
}
```

### Web

Purpose: explain web capabilities and let chat-level Tools decide whether to use them.

Do not collect web provider names, API keys, or extra web configuration in Workbench.

The current OpenAPI exposes:

- `POST /v1/tools/web-search`
- `POST /v1/tools/web-fetch`

The web search provider and its credentials belong in PrivateGPT backend config. In Workbench, show static explanatory text in Context > Web and let the chat Tools menu decide whether `web_search` and `web_extract` are included in a chat request. The direct diagnostic endpoint may still be named `/v1/tools/web-fetch`; for `/v1/messages`, use the chat tool spec `{ "name": "web_extract", "type": "web_extract_v1" }`.

### MCP

Purpose: configure MCP connectors.

Fields:

- `id`
- `name`
- `server_config_json`
- `allowed_tools`, optional list

The OpenAPI supports `mcp_servers` on `ChatBody`. The UI should allow raw JSON editing initially to avoid over-designing unknown MCP variants.

Example UI:

- Name input
- JSON textarea
- Validate JSON button

### Skills

Purpose: configure available skills.

Use the backend skills API scoped to the single active Workbench collection.

Fields:

- `id`
- `display_title`
- `collection`
- `latest_version`
- `source`
- `loading`
- `readonly`

Operations:

- `GET /v1/skills?collection=<collection>` to list skills for the active collection.
- `POST /v1/skills` multipart create for new skills.
- `POST /v1/skills/{skill_id}/versions` multipart create for new versions.
- `DELETE /v1/skills/{skill_id}?collection=<collection>` for non-readonly skills.

When chat requests are built, selected skills should be represented as a `tool_context` artifact:

```json
{
  "type": "skill",
  "skill_filter": {
    "collection": "<configured-collection>",
    "skill_or_version_ids": ["<selected-skill-id>"]
  }
}
```

### Custom Tools

Purpose: let users define Claude-style custom tools and browser-executed JavaScript handlers.

Fields:

- `id`
- `name`
- `description`
- `input_schema_json`
- `javascript_handler`
- `test_input_json`
- `last_test_result`

Tool definition shape:

```json
{
  "name": "currency_converter",
  "description": "Convert USD to EUR using a locally configured exchange rate.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "amount": {
        "type": "number"
      }
    },
    "required": ["amount"]
  }
}
```

Handler shape:

```js
async function handle(input, context) {
  const rate = Number(context.localStorage.getItem("usd_eur_rate") || "0.92");

  return {
    type: "text",
    text: `${input.amount} USD is approximately ${input.amount * rate} EUR.`
  };
}
```

Handler execution context:

```ts
{
  fetch: window.fetch.bind(window),
  localStorage: window.localStorage,
  privateGptBaseUrl: string,
  currentChatId: string,
  currentCollection: string,
  log: (message: string, data?: unknown) => void
}
```

Use browser execution directly. No extra sandbox is required for v1, because the user is explicitly authoring local browser code.

Custom tool test:

- Parse test input JSON.
- Execute handler.
- Show result or error.
- Add debugger event if currently inside a chat context, otherwise just show local result.

## Chat Screen

Main non-technical experience.

Header controls (in the composer toolbar below the textarea):

- Model selector — a custom glass dropdown populated from `GET /v1/models`, showing the current model name with an animated chevron. Selecting a model updates `state.selectedModel`.
- Refresh models icon button next to the model selector.
- Reasoning effort is selected inside the model dropdown rather than through a standalone Thinking button. The dropdown places searchable models on the left and a capability-aware effort rail on the right with None, Low, Medium, High, Max, and XHigh choices.
- The selected effort is stored per chat and sent to the messages API as `thinking: { enabled: Boolean(effort), type: effort }`. Unsupported effort choices are disabled using the selected model's `capabilities.effort` metadata.
- Tools button — opens the Tools menu popup with per-category toggles:
  - Documents
  - Web
  - Databases
  - MCP
  - Skills
  - Custom Tools

For databases, MCP, skills, and custom tools:

- Show configured items as selectable chips/dropdowns.
- Store selections per chat.

Message composer:

- Text input.
- Send button.
- Pressing `Enter` while focused in the composer sends the message.
- Pressing `Shift+Enter` inserts a line break.

Message rendering:

- User messages.
- Assistant messages.
- Text blocks rendered as Markdown.
- Tool use blocks rendered as collapsed details blocks.
- Tool result blocks rendered as collapsed details blocks.
- Inline `<citation ...></citation>` tags must never be rendered as text.
- Inline citation tags must be replaced with small circular clickable citation markers, labelled only with the citation index. If the raw citation tag uses a zero-based `index` attribute, display `index + 1`.
- Multiple inline citations may display the same number when they reference the same citation index.
- Do not render a separate citation list at the bottom of the message.
- Clicking a citation marker opens a closable glass-style popup with citation metadata and the source excerpt text from the original document when available.
- For semantic-search tool results, extract the source excerpt from the `tool_result` payload by parsing the JSON text, matching the inline citation id against `nodes[].id`, and using the matched node's `content` field as the excerpt. The matcher should tolerate bracket differences such as `4C40` vs `[4C40]`.
- Errors clearly shown.
- Do not show a raw response block in Chat; raw request/response details belong in Debugger.
- While waiting for PrivateGPT, show an in-chat pending response indicator using the PrivateGPT circular avatar with a subtle breathing animation.

Request behavior:

- Use `POST /v1/messages`.
- Build `ChatBody` from chat messages plus chat-selected context/tools.
- Use the selected model id from the `/v1/models` response. If models have not been loaded yet, fall back to `default`.
- Prefer the user-configured Settings system prompt when present.
- If no system prompt or skill instructions are present, omit system prompt text.
- When Documents are enabled and Settings > Use citations is on, send `system.citations.enabled: true` so semantic-search answers can include citation tags. This may require a top-level `system` object even when no prompt text is configured.
- Always send system instructions through the top-level `system` field so they are applied consistently across the whole request.

Basic request body:

```json
{
  "model": "default",
  "messages": [
    {
      "role": "user",
      "content": "Summarize my documents and cite sources."
    }
  ],
  "system": {
    "text": "You are a support agent. Reply with only a short ticket title.",
    "use_default_prompt": false,
    "citations": {
      "enabled": true
    }
  },
  "tools": [],
  "tool_context": [],
  "mcp_servers": [],
  "stream": false,
  "max_tokens": 4096
}
```

Tool/context building:

- If Documents are enabled, add the semantic search tool and an `ingested_artifact` tool context scoped to the global Context Documents collection. Use `artifacts: []` to search all documents in that collection.
- If Databases are selected, add selected SQL database artifacts to `tool_context`.
- If MCP connectors are selected, add selected MCP server configs to `mcp_servers`.
- If Custom Tools are selected, add selected custom tool definitions to `tools`.
- If Skills are selected, add skill prompt/config support according to current backend conventions.

### Custom Tool Loop

If assistant response contains a `tool_use` block for a custom browser tool:

1. Find matching custom tool by name.
2. Execute the JavaScript handler with `toolUse.input`.
3. Append the tool result to the same visible assistant message bubble (not as a separate message).
4. Send a follow-up `POST /v1/messages` with the tool result in the API history.
5. Stream the final assistant answer into the same bubble.
6. API calls made during the loop appear in API Debugger.

The entire tool execution cycle — initial response, tool result, and follow-up answer — appears as a single consolidated assistant message bubble in chat. Hidden messages carrying tool roles exist in the API history only and are never rendered in the chat UI.

Follow-up messages should preserve the prior conversation and include the tool result using the API's expected content block shape.

If the handler fails:

- Show the error inline in the chat bubble.
- Log request/response errors in Debugger.
- Send a tool result with `is_error: true` if appropriate.

## URL Hash Navigation

The app uses hash-based navigation so that reloading the page restores the current view and context tab.

Hash format:

- `#context/{tab}` — Context screen with a specific tab (`documents`, `databases`, `web`, `mcp`, `skills`, `customTools`).
- `#settings` — Settings screen.
- `#apiDebugger` — API Debugger screen.
- `#chat/{chatId}` — Specific chat session by ID.

`syncHash()` is called at the end of every `render()` and after context tab changes. `restoreFromHash()` runs once at startup before the first render and on `hashchange` for browser back/forward support.

## API Debugger Screen

API Debugger is session-level, live-only, and ephemeral.

Do not persist debugger events.

Show a small non-intrusive callout near the top of API Debugger explaining that it is a live trace for the current session and clears on page refresh.

Purpose:

- Teach developers how chat interactions map to API calls.
- Show request/response payloads and errors.
- Show redacted request headers when useful.

Debugger layout:

```text
Timeline list | Event detail panel
```

Event model:

- One timeline entry per API call.
- The entry may start as pending, then update in place with the response or error.
- Do not show separate request and response entries for the same API call.

Each API event should include:

- timestamp
- method
- URL
- redacted request headers
- status
- duration
- request JSON
- response JSON or error

Secrets must never be displayed in Debugger. Redact `Authorization`, token-like headers, API keys, and cookies.

API Debugger should show API events from the current page lifetime, including calls made from Chat, Context, and Settings.

On reload, debugger is empty.

## API Client

Implement a tiny client wrapper:

```ts
async function apiFetch(path, options, debugMeta)
```

Responsibilities:

- Prefix with `privateGptBaseUrl`.
- Set JSON headers.
- Build `Authorization: Basic <base64(username:password)>` from the configured username and password when provided.
- Measure duration.
- Parse JSON/text response.
- Log request/response/error to the session-level API Debugger buffer.
- Throw useful errors.

Endpoints used from OpenAPI:

- `GET /v1/models`
- `POST /v1/messages`
- `POST /v1/messages/count_tokens`, optional
- `POST /v1/messages/validate`, optional
- `POST /v1/artifacts/ingest`
- `GET /v1/artifacts/list?collection=<collection>`
- `POST /v1/artifacts/delete`
- `POST /v1/artifacts/content`, optional
- `POST /v1/tools/semantic-search`
- `POST /v1/tools/tabular-data-analysis`, optional direct diagnostic
- `POST /v1/tools/database-query`, optional direct diagnostic
- `POST /v1/tools/web-search`, optional direct diagnostic
- `POST /v1/tools/web-fetch`, optional direct diagnostic

Streaming/async endpoints can be deferred:

- `/v1/messages/async`
- `/v1/messages/async/{message_id}/stream`

## UX Principles

- The app should feel like a practical local assistant, not a marketing page.
- Chat is the primary surface.
- Context explains what the assistant can access.
- Debugger explains what happened under the hood.
- Keep controls dense but readable.
- Avoid large decorative cards or landing-page hero sections.
- Use plain, utilitarian UI.
- Prefer native controls and simple CSS.
- All errors should be visible and actionable.

## MVP Acceptance Criteria

1. User can configure API base URL.
2. User can configure an optional API key / bearer token from the UI.
3. User can load models from `GET /v1/models` and select one for each chat.
4. User can create, rename, delete, and switch local chat sessions.
5. Chat sessions persist across reload.
6. Chat-specific tool toggles persist across reload.
7. User can send a basic chat message to `/v1/messages`.
8. User can upload and ingest documents.
9. User can list ingested documents.
10. User can enable document context for chat.
11. Document-enabled chat requests add the `semantic_search` tool and an `ingested_artifact` tool context scoped to the configured Documents collection.
12. User can configure database artifacts locally.
13. User can configure MCP, skills, and custom tools locally, and can see that web provider credentials are configured in the PrivateGPT backend rather than in Workbench.
14. User can define a custom tool with name, description, JSON schema, and JavaScript handler.
15. Chat can pass selected custom tools to the API.
16. Chat can execute a browser JavaScript handler when the assistant emits a matching tool call.
17. API Debugger shows live API request/response/error events for the current browser session, with sensitive headers redacted.
18. API Debugger events disappear after browser reload.
19. No backend storage is introduced for the UI.
20. App works as a static file.
21. The implementation references the repository's relative OpenAPI file as the API contract and does not hardcode payload assumptions that contradict the schema.
22. Sidebar includes a GitHub repository widget and a Not for Production disclosure.
23. Settings includes a Clear local data action for this Workbench's browser state.

## Suggested Build Order

1. Static layout: sidebar, Context screen, Chat screen, API Debugger screen.
2. `localStorage` state model.
3. API base URL and model loading.
4. Chat sessions.
5. Basic `/v1/messages` chat.
6. Debugger event recorder.
7. Documents ingestion/list/delete.
8. Chat tool/context toggles.
9. Database/Web/MCP/Skills config forms.
10. Custom tool definition UI.
11. Browser JavaScript handler execution loop.
12. Polish errors, empty states, and reload behavior.
