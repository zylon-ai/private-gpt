from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from private_gpt.components.embedding.discovery import get_embedding_models
from private_gpt.components.llm.discovery import get_models
from private_gpt.components.model_discovery.service import are_distinct_api_bases
from private_gpt.constants import PROJECT_ROOT_PATH

if TYPE_CHECKING:
    from private_gpt.settings.settings import EmbeddingModelConfig, LLMModelConfig


class _SettingsDumper(yaml.SafeDumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        super().increase_indent(flow, False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover remote OpenAI-compatible models and write a settings profile."
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT_PATH / "settings-model.yaml"),
        help="Output settings YAML path.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout for discovery requests.",
    )
    parser.add_argument(
        "--no-fetch-all-pages",
        action="store_true",
        help="Only fetch the first page returned by the discovery endpoints.",
    )
    parser.add_argument(
        "--llm-default-model",
        default=None,
        help="Default LLM model to write. Must be one of the discovered LLM models.",
    )
    parser.add_argument(
        "--embedding-default-model",
        default=None,
        help=(
            "Default embedding model to write. Must be one of the discovered "
            "embedding models."
        ),
    )
    return parser.parse_args()


def _model_to_settings_dict(
    model: LLMModelConfig | EmbeddingModelConfig,
) -> dict[str, Any]:
    data = model.model_dump(mode="json", exclude_none=True)
    if data.get("type") == "llm" and data.get("provider") == "openai":
        data.pop("sampling_params", None)
        data.pop("reasoning_sampling_params", None)
    tags = data.get("tags")
    if isinstance(tags, list):
        data["tags"] = sorted(tags)
    return data


def _resolve_default_model(
    requested_default: str | None,
    configured_default: str,
    discovered_models: list[LLMModelConfig] | list[EmbeddingModelConfig],
    model_type: str,
) -> str:
    discovered_names = {model.name for model in discovered_models}
    if requested_default:
        if requested_default not in discovered_names:
            available = ", ".join(sorted(discovered_names)) or "none"
            raise ValueError(
                f"Unknown default {model_type} model '{requested_default}'. "
                f"Available discovered models: {available}"
            )
        return requested_default

    if configured_default and configured_default in discovered_names:
        return configured_default
    if discovered_models:
        return discovered_models[0].name
    return ""


def _find_model(
    models: list[EmbeddingModelConfig],
    name: str,
) -> EmbeddingModelConfig | None:
    return next((model for model in models if model.name == name), None)


def _write_settings_profile(
    out_path: Path,
    llm_models: list[LLMModelConfig],
    embedding_models: list[EmbeddingModelConfig],
    *,
    llm_requested_default_model: str | None,
    embedding_requested_default_model: str | None,
    llm_default_model: str,
    embedding_default_model: str,
) -> None:
    llm_default = _resolve_default_model(
        llm_requested_default_model,
        llm_default_model,
        llm_models,
        "LLM",
    )
    embedding_default = _resolve_default_model(
        embedding_requested_default_model,
        embedding_default_model,
        embedding_models,
        "embedding",
    )
    default_embedding_model = _find_model(embedding_models, embedding_default)
    output = {
        "llm": {
            "auto_discover_models": False,
            "default_model": llm_default,
        },
        "embedding": {
            "auto_discover_models": False,
            "default_model": embedding_default,
        },
        "models": [
            *[_model_to_settings_dict(model) for model in llm_models],
            *[_model_to_settings_dict(model) for model in embedding_models],
        ],
    }
    if default_embedding_model and default_embedding_model.embed_dim:
        output["vectorstore"] = {"embed_dim": default_embedding_model.embed_dim}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as file:
        yaml.dump(output, file, Dumper=_SettingsDumper, sort_keys=False)


def main() -> None:
    args = _parse_args()

    from private_gpt.settings.settings import unsafe_typed_settings

    settings = unsafe_typed_settings

    fetch_all_pages = not args.no_fetch_all_pages
    split_model_endpoints = are_distinct_api_bases(
        settings.openai.api_base,
        settings.openai.embedding_api_base,
    )
    llm_models = get_models(
        settings.openai.api_base,
        settings.openai.api_key,
        timeout=args.timeout,
        fetch_all_pages=fetch_all_pages,
        force_model_kind=split_model_endpoints,
    )
    embedding_models = get_embedding_models(
        settings.openai.embedding_api_base or settings.openai.api_base,
        settings.openai.embedding_api_key or settings.openai.api_key,
        timeout=args.timeout,
        fetch_all_pages=fetch_all_pages,
        force_model_kind=split_model_endpoints,
    )

    out_path = Path(args.out)
    _write_settings_profile(
        out_path,
        llm_models,
        embedding_models,
        llm_requested_default_model=args.llm_default_model,
        embedding_requested_default_model=args.embedding_default_model,
        llm_default_model=settings.llm.default_model,
        embedding_default_model=settings.embedding.default_model,
    )
    print(
        f"Wrote {out_path} with {len(llm_models)} LLM models and "
        f"{len(embedding_models)} embedding models."
    )


if __name__ == "__main__":
    main()
