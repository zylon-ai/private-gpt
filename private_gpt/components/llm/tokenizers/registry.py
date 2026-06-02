from collections.abc import Callable
from typing import Any

from private_gpt.components.llm.tokenizers.models.auto_discovery import (
    auto_discover_model,
)
from private_gpt.components.llm.tokenizers.tokenizer_base import TokenizerBase

TokenizerProvider = Callable[..., TokenizerBase]

_EXTERNAL_TOKENIZER_FACTORIES: dict[str, TokenizerProvider] = {}


def register_tokenizer_factory(
    tokenizer_mode: str,
    factory: TokenizerProvider,
) -> None:
    _EXTERNAL_TOKENIZER_FACTORIES[tokenizer_mode] = factory


def _build_mistral_tokenizer(**kwargs: Any) -> TokenizerBase:
    from private_gpt.components.llm.tokenizers.mistral import MistralTokenizer

    if not MistralTokenizer.is_available(**kwargs):
        raise ValueError(
            "MistralTokenizer is not available with the given configuration."
        )

    return MistralTokenizer.from_pretrained(**kwargs)


def _build_tiktoken_tokenizer(**kwargs: Any) -> TokenizerBase:
    from private_gpt.components.llm.tokenizers.tiktoken import TikTokenTokenizer

    return TikTokenTokenizer.from_pretrained(**kwargs)


def _build_estimator_tokenizer(**kwargs: Any) -> TokenizerBase:
    from private_gpt.components.llm.tokenizers.estimator import EstimatorTokenizer

    return EstimatorTokenizer.from_pretrained(**kwargs)


def _build_remote_tokenizer(**kwargs: Any) -> TokenizerBase:
    from private_gpt.components.llm.tokenizers.remote import RemoteTokenizeTokenizer

    if not RemoteTokenizeTokenizer.is_available(**kwargs):
        raise ValueError(
            "RemoteTokenizeTokenizer is not available with the given configuration."
        )

    return RemoteTokenizeTokenizer.from_pretrained(**kwargs)


def _build_huggingface_tokenizer(**kwargs: Any) -> TokenizerBase:
    from private_gpt.components.llm.tokenizers.huggingface import HuggingFaceTokenizer

    if not HuggingFaceTokenizer.is_available(**kwargs):
        raise ImportError(
            "HuggingFaceTokenizer is not available with the given configuration."
        )

    return HuggingFaceTokenizer.from_pretrained(**kwargs)


def _build_default_tokenizer(**kwargs: Any) -> TokenizerBase:
    # 1. HF as initial tokenizer
    try:
        return _build_huggingface_tokenizer(**kwargs)
    except (ImportError, Exception):
        pass

    # 2. Use the remote tokenizer
    try:
        return _build_remote_tokenizer(**kwargs)
    except (ImportError, ValueError, Exception):
        pass

    # 3. Fallback: estimate the tokenizer
    return _build_estimator_tokenizer(**kwargs)


_BUILTIN_TOKENIZER_FACTORIES: dict[str, TokenizerProvider] = {
    "mistral": _build_mistral_tokenizer,
    "remote": _build_remote_tokenizer,
    "tiktoken": _build_tiktoken_tokenizer,
    "estimator": _build_estimator_tokenizer,
    "huggingface": _build_huggingface_tokenizer,
    "chat": _build_huggingface_tokenizer,
    "default": _build_default_tokenizer,
}


class TokenizerRegistry:
    @staticmethod
    @auto_discover_model(
        enabled=True,
        tokenizer_only=True,
        raise_on_error=False,
    )
    def get_tokenizer(
        tokenizer_mode: str,
        **kwargs: Any,
    ) -> TokenizerBase:
        factory = _EXTERNAL_TOKENIZER_FACTORIES.get(
            tokenizer_mode
        ) or _BUILTIN_TOKENIZER_FACTORIES.get(tokenizer_mode)
        if factory is None:
            raise ValueError(f"Tokenizer mode {tokenizer_mode} not found.")
        return factory(**kwargs)
