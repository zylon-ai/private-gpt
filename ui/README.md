# PrivateGPT Workbench

PrivateGPT Workbench is a lightweight static demo UI for the PrivateGPT API. It lives in `./ui`, keeps the implementation in a single `index.html`, and stores supporting documentation separately so the runtime file stays easy to inspect.

## Layout

- `index.html`: the only runtime implementation file.
- `AGENTS.md`: Codex and OpenAI-style agent workflow notes.
- `CLAUDE.md`: Claude Code workflow notes.
- `docs/PRD.md`: product behavior and requirements.
- `docs/STYLE_GUIDE.md`: visual and interaction direction.
- `docs/SOURCE_OF_TRUTH.md`: canonical paths, API contract, and doc ownership.
- `references/*`: visual reference images used by the style guide.

## Working Rules

- Keep the app implementation in `index.html` unless a separate refactor is explicitly requested.
- Keep product, design, and architecture guidance in `docs/`, not in the top level beside runtime code.
- Keep visual reference images in `references/`, not mixed with the implementation file.
- Keep docs and implementation aligned whenever behavior, visuals, persistence, or API wiring changes.

## Validation

For changes to `./ui/index.html`, validate that the inline script parses:

```sh
node -e "const fs=require('fs'); const html=fs.readFileSync('./ui/index.html','utf8'); const m=html.match(/<script>([\s\S]*)<\/script>/); if(!m) throw new Error('script tag not found'); new Function(m[1]); console.log('script ok')"
```

Behaviour of the chat controls is covered by `tests/ui/`, which lifts named functions out of the inline script and runs them under Node against a small fake DOM. It needs `node` on the `PATH` and is skipped without it:

```sh
PGPT_HOME=. PYTHONPATH=. uv run pytest tests/ui
```

When a test needs a function that is not yet covered, pass its name to `run()` in `tests/ui/workbench_script.py` together with any helpers it calls; everything else in the script stays unevaluated.
