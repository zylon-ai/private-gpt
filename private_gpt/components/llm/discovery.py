from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from private_gpt.components.llm.prompt_helper import get_tokenizer
from private_gpt.components.model_discovery.client import positive_int
from private_gpt.components.model_discovery.models import ModelKind
from private_gpt.components.model_discovery.service import discover_model_infos

if TYPE_CHECKING:
    from private_gpt.chat.input_models import ModelInfoOutput
    from private_gpt.settings.settings import LLMModelConfig

DEFAULT_MODELS_VALUES: dict[str, Any] = {
    "mode": "openai",
    "prompt_style": "chat",
    "tokenizer_mode": "default",
    "enabled": True,
    "context_window": 128_000,
    "support_image": 1,
    "support_audio": 0,
    "support_tools": True,
    "support_reasoning": True,
    "api_type": "chat_completions",
}

DEFAULT_DISCOVERY_TIMEOUT = 3.0

logger = logging.getLogger(__name__)


def _capability_supported(capability: Any, default: bool) -> bool:
    if capability is None:
        return default

    supported = getattr(capability, "supported", None)
    if supported is not None:
        return bool(supported)

    if isinstance(capability, Mapping):
        return bool(capability.get("supported", default))

    return bool(capability)


def _capability_count(capability: Any, default_supported: bool) -> int | None:
    if not _capability_supported(capability, default_supported):
        return None

    maximum = getattr(capability, "maximum", None)
    if isinstance(capability, Mapping):
        maximum = capability.get("maximum", maximum)

    return positive_int(maximum) or 1


def _check_tokenizer_mode(
    model_info: ModelInfoOutput, tokenizer_mode: str, **kwargs: Any
) -> bool:
    try:
        tokenizer = get_tokenizer(tokenizer_mode, model_info.id, **kwargs)
        tokens = tokenizer.encode("Test tokenizer support", add_special_tokens=False)
    except Exception as exc:
        logger.warning(
            "This provider don't support %s mode for model '%s': %s",  # TODO: add link to dos
            tokenizer_mode,
            model_info.id,
            exc,
        )
        return False
    return bool(tokens)


def _probe_chat_completions_endpoint(
    api_base: str, api_key: str | None, model_id: str, timeout: float
) -> bool:
    """Return True if the chat/completions endpoint responds (non-404)."""
    import requests

    url = api_base.rstrip("/") + "/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 1,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        return resp.status_code != 404
    except Exception as exc:
        logger.debug("Chat completions probe failed for %s: %s", api_base, exc)
        # Cannot reach server — conservatively assume the endpoint exists.
        return True


def _probe_responses_endpoint(
    api_base: str, api_key: str | None, model_id: str, timeout: float
) -> bool:
    """Return True if the responses endpoint responds (non-404)."""
    import requests

    url = api_base.rstrip("/") + "/responses"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model_id,
        "input": "test",
        "max_output_tokens": 1,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        return resp.status_code != 404
    except Exception as exc:
        logger.debug("Responses API probe failed for %s: %s", api_base, exc)
        return False


def _get_openai_api_type_for_model(model_id: str) -> str:
    """Return the api_type for a real OpenAI model using llama-index's model registry.

    Models in RESPONSES_API_ONLY_MODELS (e.g. gpt-5.2-pro) must use the
    Responses API; all others default to chat_completions.
    """
    try:
        from llama_index.llms.openai.utils import (  # ty:ignore[unresolved-import]
            is_chatcomp_api_supported,
        )

        return (
            "chat_completions" if is_chatcomp_api_supported(model_id) else "responses"
        )
    except ImportError:
        return "chat_completions"


def _probe_api_type(
    model_infos: list[ModelInfoOutput],
    api_base: str,
    api_key: str | None,
    timeout: float = DEFAULT_DISCOVERY_TIMEOUT,
) -> str:
    """Detect which API type an OpenAI-compatible endpoint supports.

    Probes chat/completions first; if that returns 404, falls back to responses.
    This function is only called for non-OpenAI endpoints — real OpenAI uses
    ``_get_openai_api_type_for_model`` per model instead.
    """
    if not model_infos:
        return str(DEFAULT_MODELS_VALUES["api_type"])

    model_id = model_infos[0].id

    if _probe_chat_completions_endpoint(api_base, api_key, model_id, timeout):
        logger.debug("Detected chat_completions API type for %s", api_base)
        return "chat_completions"

    if _probe_responses_endpoint(api_base, api_key, model_id, timeout):
        logger.info(
            "chat/completions endpoint not found for %s; using Responses API", api_base
        )
        return "responses"

    logger.debug(
        "Could not detect API type for %s, defaulting to chat_completions", api_base
    )
    return str(DEFAULT_MODELS_VALUES["api_type"])


def _model_info_to_config(
    model_info: ModelInfoOutput,
    tokenizer_mode: str,
    mode: str | None = None,
    provider: str | None = None,
    api_type: str = "chat_completions",
) -> LLMModelConfig:
    from private_gpt.settings.settings import LLMModelConfig, SamplingParams

    capabilities = model_info.capabilities
    supports_reasoning = DEFAULT_MODELS_VALUES["support_reasoning"]
    supports_tools = DEFAULT_MODELS_VALUES["support_tools"]
    support_image: int | None = DEFAULT_MODELS_VALUES["support_image"]
    support_audio: int | None = DEFAULT_MODELS_VALUES["support_audio"]

    if capabilities is not None:
        supports_reasoning = _capability_supported(
            capabilities.thinking, supports_reasoning
        ) or _capability_supported(capabilities.effort, supports_reasoning)
        supports_tools = _capability_supported(
            capabilities.structured_outputs, supports_tools
        )
        support_image = _capability_count(capabilities.image_input, bool(support_image))
        support_audio = _capability_count(capabilities.audio_input, bool(support_audio))

    # Default privateGPT values
    mode = mode or DEFAULT_MODELS_VALUES["mode"]
    prompt_style = DEFAULT_MODELS_VALUES["prompt_style"]

    max_new_tokens = positive_int(model_info.max_tokens)
    sampling_params = (
        SamplingParams(max_new_tokens=max_new_tokens)
        if max_new_tokens is not None
        else SamplingParams()
    )

    return LLMModelConfig(
        name=model_info.id,
        mode=mode,
        provider=provider,
        prompt_style=prompt_style,
        tokenizer_mode=tokenizer_mode,
        enabled=True,
        alias=model_info.id,
        api_type=api_type,
        context_window=positive_int(model_info.max_input_tokens)
        or DEFAULT_MODELS_VALUES["context_window"],
        support_image=support_image,
        support_audio=support_audio,
        support_tools=supports_tools,
        support_reasoning=supports_reasoning,
        sampling_params=sampling_params,
        reasoning_sampling_params=sampling_params,
    )


def _get_tokenizer_mode(
    model_infos: list[ModelInfoOutput],
    api_base: str,
    api_key: str | None,
) -> str:
    if not model_infos:
        return str(DEFAULT_MODELS_VALUES["tokenizer_mode"])

    first_model = model_infos[0]
    if _check_tokenizer_mode(
        first_model,
        "remote",
        api_base=api_base,
        api_key=api_key,
    ):
        return "remote"

    return str(DEFAULT_MODELS_VALUES["tokenizer_mode"])


def get_models(
    api_base: str,
    api_key: str | None,
    *,
    mode: str | None = None,
    timeout: float = DEFAULT_DISCOVERY_TIMEOUT,
    fetch_all_pages: bool = True,
    force_model_kind: bool = False,
) -> list[LLMModelConfig]:
    from private_gpt.components.model_discovery.url_utils import is_openai_api_base

    discovery = discover_model_infos(
        api_base,
        api_key,
        force_kind=ModelKind.LLM if force_model_kind else None,
        timeout=timeout,
        fetch_all_pages=fetch_all_pages,
    )
    model_infos = list(discovery.llm_models)
    tokenizer_mode = _get_tokenizer_mode(
        model_infos,
        api_base=api_base,
        api_key=api_key,
    )

    # For real OpenAI, determine api_type per model (some models are responses-only).
    # For other endpoints, probe once and apply the result to all models.
    use_per_model_openai = is_openai_api_base(api_base)
    probed_api_type: str | None = None
    if not use_per_model_openai:
        probed_api_type = _probe_api_type(model_infos, api_base, api_key, timeout)

    configs: list[LLMModelConfig] = []
    for model_info in model_infos:
        api_type = (
            _get_openai_api_type_for_model(model_info.id)
            if use_per_model_openai
            else (probed_api_type or str(DEFAULT_MODELS_VALUES["api_type"]))
        )
        configs.append(
            _model_info_to_config(
                model_info,
                tokenizer_mode=tokenizer_mode,
                mode=mode,
                provider=discovery.provider.value,
                api_type=api_type,
            )
        )

    return configs
