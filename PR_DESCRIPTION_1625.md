# feat: support loading HuggingFace models from local directories

Closes #1625

## Problem
Users who download large HuggingFace models manually (e.g. `Salesforce/SFR-Embedding-Mistral`) have no way to tell private-gpt to load them from a local directory. The only option today is `embedding_hf_model_name`, which always triggers a Hub download. If the download fails or the environment is air-gapped, the application cannot start.

The same problem applies to LLM GGUF files and tokenizers — there is no local-path override for any of these.

## Solution
Add three optional settings fields that, when set, override the corresponding Hub-name settings and load models directly from disk:

- `embedding_hf_model_path` in `HuggingFaceSettings` overrides `embedding_hf_model_name` for local embedding model directories.
- `llm_local_path` in `LlamaCPPSettings` overrides `llm_hf_model_file` lookup under `models/` for local GGUF files.
- `tokenizer_path` in `LLMSettings` overrides `tokenizer` for local tokenizer directories.

All three fields:
- Default to `None` (no behavior change for existing users).
- Accept absolute paths or paths relative to project root.
- Support `~` expansion.
- Validate existence at startup and raise clear `ValueError` on invalid paths.

## How to use (example from the issue)
```bash
# 1) Download model assets locally
huggingface-cli download Salesforce/SFR-Embedding-Mistral \
  --local-dir /home/user/models/SFR-Embedding-Mistral

# 2) Configure local paths in settings.yaml or settings-local.yaml
```

```yaml
huggingface:
  embedding_hf_model_name: Salesforce/SFR-Embedding-Mistral
  embedding_hf_model_path: /home/user/models/SFR-Embedding-Mistral

llm:
  tokenizer_path: /home/user/models/tokenizer

llamacpp:
  llm_local_path: /home/user/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

## Files changed
Settings:
- `private_gpt/settings/settings.py` (new fields: `embedding_hf_model_path`, `llm_local_path`, `tokenizer_path`)

Model loading:
- `private_gpt/components/embedding/embedding_component.py` (local embedding path resolve + existence check + override)
- `private_gpt/components/llm/llm_component.py` (local tokenizer / local GGUF support + validation)

Path utilities:
- `private_gpt/paths.py` (`absolute_or_from_project_root` export + `Path.expanduser()` support)

Configuration examples:
- `settings.yaml` (commented examples for local embedding/LLM/tokenizer paths)
- `settings-local.yaml` (commented local-path examples)

Documentation:
- `fern/docs/pages/installation/troubleshooting.mdx` (manual local model section)
- `fern/docs/pages/manual/llms.mdx` (local model file section)

Tests:
- `tests/test_local_hf_model_path.py` (covers fallback behavior, local-path precedence, missing-path validation, and settings schema behavior)

## Type of Change
- [x] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [x] This change requires a documentation update

## How Has This Been Tested?
- [x] Added new unit/integration tests
- [x] I stared at the code and made sure it makes sense

Verified locally:
1. `ruff check` passes on changed files.
2. Schema validation for local path fields works.
3. Full pytest run is blocked in this environment due to missing optional plugins/dependencies (e.g. `injector`) before tests execute.

Test Configuration:
- Firmware version: N/A
- Hardware: N/A
- Toolchain: Python + pydantic + llama-index + transformers
- SDK: N/A

## Dependencies
- No new mandatory runtime dependencies introduced.
- Reuses existing project integrations and utilities.

## Risk / Impact
Low. New fields default to `None`, so existing behavior remains unchanged unless users opt in.

## Rollback Plan
If regression appears:
1. Revert settings schema additions for local-path fields.
2. Revert local-path override logic in embedding/LLM components.
3. Revert docs/settings example updates.
4. Revert issue #1625 commit(s).

## Checklist
- [x] My code follows the style guidelines of this project
- [x] I have performed a self-review of my code
- [x] I have commented my code, particularly in hard-to-understand areas
- [x] I have made corresponding changes to the documentation
- [x] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published in downstream modules
- [ ] I ran `make check; make test` to ensure mypy and tests pass
