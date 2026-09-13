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

## Client-Side Persistence

`state` is persisted to `localStorage` under `privategpt-workbench-state-v1`, and that includes
server-derived data such as `state.models`. Anything cached from the server must have a defined
refresh path, otherwise it silently goes stale and a page reload will not fix it. `state.models` is
refreshed on startup, by the refresh-models control beside the model picker, and by Settings >
Test API.

Some UI state is deliberately **not** persisted and lives on `runtime` instead, so that a reload
starts from a clean view: `runtime.stickToBottom` and `runtime.lastRenderedChatId` (transcript
scroll following), `runtime.expandedMessages` (which long messages the reader has opened),
`runtime.editingMessageId` (the message open for editing) and `runtime.abortControllers` (the
in-flight request per chat).

Persisted alongside the chats: `chat.settings.maxTokens` (per-chat output cap, null means use the
model's advertised value), `chat.contextTokens` (last known input+output usage) and
`state.sidebarCollapsed`.

## Serving Note

`index.html` is read into memory when the PrivateGPT server starts (`launcher.py` serves it via
`HTMLResponse(content=_index_html)`, not from disk per request). Edits to `index.html` therefore
require a server restart before they are served — a browser reload alone shows the old page.

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
- **Reasoning effort lives in the model selector** — each chat stores `chat.settings.reasoningEffort` as `null`, `low`, `medium`, `high`, `max`, or `xhigh`, defaulting to `null` (None). The request sends this as `thinking: { enabled, type }`, and effort options are enabled from the selected model's `capabilities.effort` response. Selecting a model or effort updates the existing popup DOM in place, preserving search and scroll position; it closes only from its trigger, Escape, or an outside click.
- **Composer attachments reuse existing upload paths** — the composer plus button sends files to the active code-execution session when Code Execution is enabled and otherwise ingests them into the configured Documents collection.
- **Composer actions are consolidated under the plus button** — the plus trigger opens one menu containing Add files followed by the existing chat tool/context controls. There is no separate Build mode or Build button.
- **Hash navigation** — `syncHash()` / `restoreFromHash()` keep the URL in sync with the active view and context tab. Format: `#context/{tab}`, `#chat/{id}`, `#settings`, `#apiDebugger`.
- **Scroll fades** — `.chat-list-wrap` and `.messages` both use `mask-image` with `--fade-top-stop`/`--fade-bot-stop` custom properties updated on scroll by `updateChatListFade()` and `updateMessagesFade()`.
- **Transcript scrolling goes through `scrollMessagesToBottom()`** — never assign `scrollTop` directly. It follows new content only while `runtime.stickToBottom` is set, which `syncStickToBottom()` derives from scroll position on every scroll event. Pass `{ force: true }` only where following the newest message is the point: sending, opening a chat, and the Jump to latest control.
- **`.messages` sits inside `.messages-wrap`** — the wrapper owns the column width and gives the Jump to latest control a positioning context, so it can float over the transcript instead of taking part in the column's vertical flow, which the composer already resizes.
- **Chat width is an appearance value** — `state.uiAppearance.chatWidth` is one of `comfortable`, `wide`, or `full`, mapped through `CHAT_WIDTHS` onto the `--chat-max` custom property by `paintAppearance()`. Like `themeMode`, it has its own control rather than a shared appearance form field, so `readAppearanceForm()` and `normalizeAppearanceSuggestion()` both carry the existing value forward — otherwise saving a palette or generating a theme would silently reset the layout.
- **Message collapse is measured, not guessed** — `applyMessageCollapse()` runs once at the end of `renderMessages()`, when nodes are in the DOM. It removes the `collapsed` class from every candidate, then reads every height, then writes the classes, so the pass costs one layout rather than one per message. Streaming messages are skipped.
- **Message actions are dispatched by class** — the delegated handler matches `.msg-action-btn` and switches on `data-action`, so a new action needs no change to the selector.
- **The message editor is built in a pass, not in the render path** — `applyMessageEditors()` runs after `renderMessages()` finishes and leaves an existing editor alone. Message nodes are rebuilt whenever their render key changes, for reasons unrelated to editing, so building the editor from the render path would discard an in-progress draft.
- **Stopping a stream** — `runtime.abortControllers` holds one `AbortController` per in-flight chat, and `apiStreamFetch` takes the signal. `requestAssistant()` treats `AbortError` as a normal ending: it keeps the partial content, writes it into `apiMessage` so the next turn can refer to it, and returns without throwing, so no caller reports a failure. An abort with no content removes the empty message. The controller is always cleared in a `finally`.
- **The send button doubles as Stop** — `renderSendButton()` owns its label, class and disabled state. Stop must stay enabled when no model is selected, or a request started before the model list changed could not be cancelled.
- **Token counting** — the stream's `usage` (`message_start` for input, `message_delta` for output) is captured in `apiStreamFetch` and stored as `chat.contextTokens`. `POST /v1/messages/count_tokens` gives a more accurate figure for the *next* request but **rejects an empty conversation** with "Messages cannot be empty", so the meter is disabled until `sanitizeMessages()` returns something.
- **The input context window is server-side** — `ChatBody` has `max_tokens` (output) but no input-window field. The window comes from the model's `context_window` in the server's settings and needs a restart; only `max_tokens` is settable from here.
- **Toggle switches** — all `input[type="checkbox"]` elements are styled as custom CSS pill toggles with no native appearance.
- **Floating panel frost** — `.modal-card`, `.menu-panel`, and `.model-dropdown` override the shared glass group with a near-solid dark background (`rgba(10,12,22,0.82–0.94)`), `blur(72px) saturate(1.4)`, and a `to bottom` gradient (lighter at top, denser at bottom) for readability and visual grounding.
- **Code Execution tools** — the Code Execution toggle in the Tools menu sends `{ name: "code_execution", type: "code_execution_v1" }` in the tools array. The backend expands this into `bash`, `text_editor` (view/str_replace/create/insert), `present_files`, and `present_server`. Tool use blocks for these tools render as styled `.code-exec-block` details elements with terminal output, line-numbered file views, diff highlighting, and exit-code badges. Adjacent tool_use + tool_result blocks are combined into a single block via blocks pairing in `blocksToHtml`. The `isCodeExecTool` allowlist (`bash`, `view`, `str_replace`, `create`, `insert`, `present_files`, `present_server`) drives pairing and per-tool rendering. The toggle is stored per-chat in `chat.settings.enabledCodeExecution`.
- **Code Execution session continuity** — when code execution is enabled, `chat.id` is sent as `container` in `ChatBody` so the backend reuses the same sandbox session across all messages in a chat. The `container` field must be set whenever code execution tools are active.
- **Code Execution file upload** — when Code Execution is enabled, a **Files** button appears in the composer toolbar. It opens a file picker that uploads directly to `POST /v1/files?scope_id={chat.id}` (multipart/form-data). Uploaded files land in the session workspace and are accessible to the model's bash/file tools.
- **Code Execution file downloads** — `present_files` tool results that contain `local_resource` blocks are rendered as `.code-exec-download` anchor elements linking to `GET /v1/files/{file_id}/content?scope_id={chat.id}`. The `file_id` and `mime_type` come from the `local_resource` block schema.
- **Code Execution server links** — `present_server` tool results that contain `resource_link` blocks are rendered as `.code-exec-server-link` anchor elements (globe icon + service name + tunneled URL) opening in a new tab. The `uri`, `name`, and `description` come from the `resource_link` block schema. The tool_use summary shows `service_name:port` (and an optional `initial_path` deep-link).
