import importlib
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from llama_index.llms.openai_like import (  # type: ignore[import-not-found,import-untyped]
        OpenAILikeResponses as OpenAILikeResponsesBase,
    )

    from private_gpt.components.llm.custom.openairesponses import (
        PatchedOpenAIResponsesLLM,
    )


def _load_openai_like_responses_base() -> type[Any]:
    try:
        return cast(
            type[Any],
            importlib.import_module("llama_index.llms.openai_like").OpenAILikeResponses,
        )
    except (ImportError, AttributeError) as e:
        from private_gpt.utils.dependencies import format_missing_dependency_message

        raise ImportError(
            format_missing_dependency_message(
                "OpenAI-like Responses LLM",
                extras="llm-openai-compatible",
            )
        ) from e


def _load_patched_openai_responses() -> type[Any]:
    from private_gpt.components.llm.custom.openairesponses import (
        PatchedOpenAIResponsesLLM,
    )

    return PatchedOpenAIResponsesLLM


if not TYPE_CHECKING:
    OpenAILikeResponsesBase = _load_openai_like_responses_base()
    PatchedOpenAIResponsesLLM = _load_patched_openai_responses()


class PatchedOpenAILikeResponsesLLM(  # ty: ignore[inconsistent-mro]
    OpenAILikeResponsesBase, PatchedOpenAIResponsesLLM
):  # type: ignore[misc]
    """OpenAILikeResponses with all PatchedOpenAIResponsesLLM fixes applied.

    OpenAILikeResponses already inherits from OpenAIResponses, so MRO ensures
    OpenAILikeResponses fields (context_window, is_function_calling_model, tokenizer)
    take precedence while all method patches from PatchedOpenAIResponsesLLM are
    inherited unchanged.
    """
