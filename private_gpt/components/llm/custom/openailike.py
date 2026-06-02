import importlib
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from llama_index.llms.openai_like import (  # type: ignore[import-not-found,import-untyped]
        OpenAILike as OpenAILikeBase,
    )

    from private_gpt.components.llm.custom.openai import PatchedOpenAILLM


def _load_openai_like_base() -> type[Any]:
    try:
        return cast(
            type[Any],
            importlib.import_module("llama_index.llms.openai_like").OpenAILike,
        )
    except ImportError as e:
        from private_gpt.utils.dependencies import format_missing_dependency_message

        raise ImportError(
            format_missing_dependency_message(
                "OpenAI-like LLM",
                extras="llm-openai-compatible",
            )
        ) from e


def _load_patched_openai() -> type[Any]:
    from private_gpt.components.llm.custom.openai import PatchedOpenAILLM

    return PatchedOpenAILLM


if not TYPE_CHECKING:
    OpenAILikeBase = _load_openai_like_base()
    PatchedOpenAILLM = _load_patched_openai()


class PatchedOpenAILikeLLM(OpenAILikeBase, PatchedOpenAILLM):  # type: ignore[misc]
    """OpenAILike with all PatchedOpenAILLM fixes applied.

    OpenAILike already inherits from OpenAI, so MRO ensures OpenAILike fields
    (is_chat_model, api_version, etc.) take precedence while all method patches
    from PatchedOpenAILLM are inherited unchanged.
    """
