import functools
import logging
import os
import sys
import typing
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic.v1.utils import deep_update, unique_list

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.settings.yaml import load_yaml_with_envvars

logger = logging.getLogger(__name__)

_settings_folder_str = os.environ.get("PGPT_SETTINGS_FOLDER", str(PROJECT_ROOT_PATH))
_settings_folders = unique_list(
    [item.strip() for item in _settings_folder_str.split(",") if item.strip()]
)

# if running in unittest, use the test profile
_test_profile = ["test"] if "tests.fixtures" in sys.modules else []

active_profiles: list[str] = unique_list(
    ["default"]
    # try to load override profile if the file exist
    + (
        ["override"]
        if any(
            (Path(folder) / "settings.override.yaml").is_file()
            for folder in _settings_folders
        )
        else []
    )
    + [
        item.strip()
        for item in os.environ.get("PGPT_PROFILES", "").split(",")
        if item.strip()
    ]
    + _test_profile
)


def merge_settings(settings: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return functools.reduce(deep_update, settings, {})


def load_settings_from_profile(profile: str) -> dict[str, Any]:
    if profile == "default":
        profile_file_name = "settings.yaml"
    elif profile == "override":
        profile_file_name = "settings.override.yaml"
    else:
        profile_file_name = f"settings-{profile}.yaml"

    config: dict[str, Any] = {}
    found = False
    for settings_folder in _settings_folders:
        path = Path(settings_folder) / profile_file_name
        if not path.is_file():
            continue
        with Path(path).open("r") as f:
            raw = load_yaml_with_envvars(f)
        if not isinstance(raw, dict):
            raise TypeError(f"Config file has no top-level mapping: {path}")
        config = raw
        found = True
        break

    if not found:
        raise FileNotFoundError(
            f"Settings file not found for profile '{profile}'. "
            f"Searched in folders: {_settings_folders} with file name '{profile_file_name}'"
        )

    return config


@typing.no_type_check
def discover_models_from_environment(
    environ: dict[str, Any] = os.environ
) -> list[dict[str, Any]]:
    """Discover model configurations from environment variables.

    This function parses environment variables with the pattern:
    PGPT_MODELS_<MODEL_ID>_<PARAMETER>[__<NESTED_PARAM>]

    Examples:
        PGPT_MODELS_GPT4_API_KEY=sk-123 ->
            {"gpt4": {"api_key": "sk-123", "name": "gpt4"}}
        PGPT_MODELS_CLAUDE_CONFIG__TEMPERATURE=0.7 ->
            {"claude": {"config": {"temperature": "0.7"}, "name": "claude"}}

    Args:
        environ: Dictionary of environment variables (defaults to os.environ)

    Returns:
        Dictionary mapping model IDs to their configuration dictionaries.
        Each model config includes a "name" field set to the model ID.

    Raises:
        ValueError: When there's a conflict between scalar and nested parameter values

    Notes:
        - Model IDs are automatically lowercased
        - Parameter names are automatically lowercased
        - Double underscores (__) create nested parameter structures
        - Each model automatically gets a "name" field with its ID
    """
    models = {}

    def _set_nested_param(
        config: dict[str, Any],
        param_keys: list[str],
        value: str,
        model_id: str,
        env_key: str,
    ) -> None:
        """Set a nested parameter value, creating intermediate dictionaries."""
        current = config

        for key in param_keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                raise ValueError(
                    f"Conflict setting environment variable for model {model_id}: {env_key}. "
                    f"Parameter '{key}' is already set as a scalar value."
                )
            current = current[key]

        current[param_keys[-1]] = value

    for key, value in environ.items():
        if key.startswith("PGPT_MODELS_") and key.count("_") >= 3:
            _, _, model_id, param = key.split("_", 3)

            model_id = model_id.lower()
            if model_id not in models:
                models[model_id] = {"name": model_id}

            param_lower = param.lower()
            param_keys = param_lower.split("__")

            _set_nested_param(models[model_id], param_keys, value, model_id, key)

    return list(models.values())


def load_active_settings() -> dict[str, Any]:
    """Load active profiles and merge them."""
    logger.info("Starting application with profiles=%s", active_profiles)
    loaded_profiles = [
        load_settings_from_profile(profile) for profile in active_profiles
    ]
    merged: dict[str, Any] = merge_settings(loaded_profiles)

    discovered_models = discover_models_from_environment()
    if discovered_models:
        merged["models"] = [
            *(merged.get("models", []) or []),
            *discovered_models,
        ]
    return merged
