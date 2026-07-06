# Source Of Truth

This document defines where the authoritative guidance for `ui/` lives.

## Runtime Implementation

- `../index.html`: the only runtime implementation file for the Workbench demo.

## Human And Agent Entry Points

- `../README.md`: top-level orientation and folder map.
- `../AGENTS.md`: Codex / OpenAI-style agent workflow notes.
- `../CLAUDE.md`: Claude Code workflow notes.

## Product And Design Docs

- `PRD.md`: product behavior and information architecture.
- `STYLE_GUIDE.md`: visual and interaction direction.

## Reference Assets

- `../references/*`: non-runtime visual references used by the style guide.

## API Contract

Workbench should follow the Fern-generated OpenAPI schema at:

```text
../../fern/openapi/openapi.json
```

Do not maintain a duplicated UI-local OpenAPI snapshot.

## Working Rules

- Product requirements belong in `docs/PRD.md`.
- Visual rules belong in `docs/STYLE_GUIDE.md`.
- Agent-specific workflow rules belong in `AGENTS.md` and `CLAUDE.md`.
- Reference images belong in `references/`.
- Runtime code belongs in `index.html`.

If a change affects behavior, visuals, persistence, security posture, or API request/response handling, update the relevant docs in the same change.

## Key Implementation Notes

These are things not obvious from reading `index.html` that future agents should know:

- **Collection lives in Settings**, not in the Documents panel. `state.context.documents.defaultCollection` is the single global collection name used by all document and chat operations.
- **Onboarding is stateful** — `state.onboarding` controls the first-run overlay, its current step, and the last live verification result. The overlay is shown whenever `state.onboarding.completed !== true`.
- **Appearance overrides are runtime variables** — `state.uiAppearance` drives copy, feature visibility, and CSS custom properties through `applyAppearance()`. Settings and onboarding write into the same structure.
- **Appearance generation uses the chat API** — the theme brief in onboarding/Settings is sent through `POST /v1/messages`, parsed as JSON, then written back into the same appearance form fields the user can edit manually.
- **Custom tool execution is consolidated** — the initial response, tool result, and follow-up answer all render inside a single assistant message bubble. Hidden messages (`hidden: true`) carry API history only and are never rendered.
- **Model selector is a custom dropdown**, not a native `<select>`. It uses `#modelSelectBtn` + `#modelDropdown`. The `renderModelSelect()` function populates it.
- **Hash navigation** — `syncHash()` / `restoreFromHash()` keep the URL in sync with the active view and context tab. Format: `#context/{tab}`, `#chat/{id}`, `#settings`, `#apiDebugger`.
- **Scroll fades** — `.chat-list-wrap` and `.messages` both use `mask-image` with `--fade-top-stop`/`--fade-bot-stop` custom properties updated on scroll by `updateChatListFade()` and `updateMessagesFade()`.
- **Toggle switches** — all `input[type="checkbox"]` elements are styled as custom CSS pill toggles with no native appearance.
- **Floating panel frost** — `.modal-card`, `.menu-panel`, and `.model-dropdown` override the shared glass group with a near-solid dark background (`rgba(10,12,22,0.82–0.94)`), `blur(72px) saturate(1.4)`, and a `to bottom` gradient (lighter at top, denser at bottom) for readability and visual grounding.
- **Code Execution tools** — the Code Execution toggle in the Tools menu sends `{ name: "code_execution", type: "code_execution_v1" }` in the tools array. The backend expands this into `bash`, `text_editor` (view/str_replace/create/insert), `present_files`, and `present_server`. Tool use blocks for these tools render as styled `.code-exec-block` details elements with terminal output, line-numbered file views, diff highlighting, and exit-code badges. Adjacent tool_use + tool_result blocks are combined into a single block via blocks pairing in `blocksToHtml`. The `isCodeExecTool` allowlist (`bash`, `view`, `str_replace`, `create`, `insert`, `present_files`, `present_server`) drives pairing and per-tool rendering. The toggle is stored per-chat in `chat.settings.enabledCodeExecution`.
- **Code Execution session continuity** — when code execution is enabled, `chat.id` is sent as `container` in `ChatBody` so the backend reuses the same sandbox session across all messages in a chat. The `container` field must be set whenever code execution tools are active.
- **Code Execution file upload** — when Code Execution is enabled, a **Files** button appears in the composer toolbar. It opens a file picker that uploads directly to `POST /v1/files?scope_id={chat.id}` (multipart/form-data). Uploaded files land in the session workspace and are accessible to the model's bash/file tools.
- **Code Execution file downloads** — `present_files` tool results that contain `local_resource` blocks are rendered as `.code-exec-download` anchor elements linking to `GET /v1/files/{file_id}/content?scope_id={chat.id}`. The `file_id` and `mime_type` come from the `local_resource` block schema.
- **Code Execution server links** — `present_server` tool results that contain `resource_link` blocks are rendered as `.code-exec-server-link` anchor elements (globe icon + service name + tunneled URL) opening in a new tab. The `uri`, `name`, and `description` come from the `resource_link` block schema. The tool_use summary shows `service_name:port` (and an optional `initial_path` deep-link).
