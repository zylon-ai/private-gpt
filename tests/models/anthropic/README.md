# Anthropic model compatibility tests

These tests pin PrivateGPT's Anthropic-compatible request/response models against:

1. The installed `anthropic` Python SDK (`tests/models/anthropic/registry.py` + `test_field_parity.py`).
2. Anthropic's published OpenAPI spec (`test_openapi_schema.py` + `openapi_drift_whitelist.json`).

Local schema may *extend* Anthropic. Remote-only differences must be listed in
`openapi_drift_whitelist.json` with an explicit reason. When the spec URL or SDK
version changes, re-run the OpenAPI tests and update the whitelist — do not
silence new diffs without documenting them here.

## Current pending gaps (anthropic 1.2.0)

The dependabot bump `anthropic 0.122.0 → 1.2.0` (and the matching OpenAPI spec
snapshot) introduced several real Anthropic features that we do **not** model
yet. They are intentionally left as SDK-only / remote-only drift until we add
end-to-end support.

### `toolset_name` on `tool_use` / `tool_result`

SDK `ToolUseBlock` and `ToolResultBlockParam` now carry optional
`toolset_name: str | None` ("for a toolset member, the toolset family").

Our `ToolUseBlock` / `ToolResultBlock` do not expose it. The field is listed in
`TypeMapping.sdk_only_fields` for `ToolUseBlock` and allowed as remote-only
OpenAPI drift. Clients sending `toolset_name` will have it dropped on parse
(`extra="allow"` on content blocks does *not* round-trip unknown fields unless
the model declares them).

**To support:** add `toolset_name: str | None = None` to both blocks, thread it
through serialization, and drop it from `sdk_only_fields` / the whitelist.

### File-based image / document sources (`type: "file"`)

Anthropic added `FileImageSourceParam` / `FileDocumentSourceParam`:

```json
{"type": "file", "file_id": "file_01abc"}
```

as an alternative to `base64` / `url` sources on `image` and `document` blocks
(including nested document content and web-fetch results).

Our `ImageSource` / `DocumentBlock.source` unions only cover `base64`, `url`,
`text`, and `content`. File sources are remote-only in the OpenAPI fingerprint.

**To support:** add a `FileSource` (`type: Literal["file"]`, `file_id: str`) to
both unions and resolve `file_id` through our Files API
(`private_gpt/server/files/`) when converting to LlamaIndex / sending to the
model. We already issue `file_id`s for sandbox outputs; this is the inverse
path (consume an uploaded file as an image/document source).

### Image `transformations`

`image` blocks now accept:

```json
{"transformations": {"oversized_image": "downsize" | "error"}}
```

Default upstream is `"downsize"`. We do not declare `transformations` on
`ImageBlock`, so the field is dropped.

**To support:** add `ImageTransformations` to `ImageBlock` and honor
`oversized_image="error"` in `MessageInput._extract_content` (we already reject
images over `settings.chat.maximum_blob_size`; this should become the
Anthropic-shaped switch).

### `browser_state` inside `tool_result`

New `tool_result` content variant used by Anthropic's browser toolset: a full
tab inventory plus `state_changes` (tab opened, download started/completed/failed).
We have no browser toolset and no `BrowserStateBlock`.

**To support:** model `BrowserStateBlock` (tabs + discriminated `state_changes`)
and add it to `ToolResultContentBlockType`. This is a larger feature, not a
passthrough field.

### `container.skills`

Anthropic's `CreateMessageParams.container` is now
`string | {id, skills[]} | null`, where `skills[]` lists
`{skill_id, type: "anthropic"|"custom", version?}`.

`ChatBody.container` accepts the string and the object (including `skills` and
any future extra keys). **Only `id` is consumed** (`resolve_container_id`) and
forwarded as today's string session id. `skills` is parsed onto the request
model but not loaded into `tool_context`.

**To support:** map `container.skills` onto `SkillArtifact` / `SkillFilter` and
merge into `tool_context`. Blocked on a collection/tenant default for
`SkillFilter.collection` — Anthropic's payload has no collection.

### Legacy Text Completions SDK types

SDK 1.x removed `anthropic.types.Completion` / `client.completions.create()`
(`/v1/complete`). We still expose `CompletionInput` / `CompletionOutput` and
validate samples against the OpenAPI spec, but no longer cross-check a SDK
Pydantic model (the sample's `sdk_model` is `None`).

**To support:** either keep the Zylon `/complete` surface indefinitely, or
deprecate it now that Anthropic has fully removed the client.

## Updating the whitelist

`openapi_drift_whitelist.json` is the contract for allowed remote-only diffs.

When `test_fastapi_openapi_contract_drift_fingerprint` fails:

1. Inspect `unexpected_remote` (new Anthropic shape we don't have) vs
   `missing_remote` (whitelist entries that no longer apply).
2. If we **intentionally** don't support the new shape yet, add the exact diff
   key with a reason and a note in this README.
3. If the remote spec now matches us, **delete** the stale whitelist entry
   (`missing_remote`).
4. If we added the field/model, the unexpected diff should disappear — do not
   keep a whitelist entry for something we now implement.
