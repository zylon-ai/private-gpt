# Codex Instructions

This file is for Codex and other OpenAI-style coding agents working in `./ui`.

## Read First

Read these files in order before changing behavior, UI, or API wiring:

1. `README.md`
2. `docs/SOURCE_OF_TRUTH.md`
3. `docs/PRD.md`
4. `docs/STYLE_GUIDE.md`
5. `index.html`

## Scope

- Keep the app as a simple static demo.
- Keep `index.html` as the only runtime implementation file unless the user explicitly asks for a different structure.
- Treat `docs/SOURCE_OF_TRUTH.md` as the canonical pointer to API contract paths and documentation ownership.
- Update the relevant docs in `docs/` whenever behavior, visuals, persistence, security posture, or API request/response handling changes.

## Codex-Specific Notes

- Follow the validation steps documented in `README.md` and `docs/SOURCE_OF_TRUTH.md`.
- If implementation and docs disagree, fix the disagreement in the same change.
- When finishing work, summarize what changed, what validation ran, and any known limitations.
