import logging
import warnings
from typing import Any

from private_gpt.components.llm.prompt_styles.prompt_style_base import PromptStyleBase
from private_gpt.components.llm.tokenizers.tokenizer_base import TokenizerBase

logger = logging.getLogger(__name__)


def get_tokenizer(
    tokenizer_mode: str,
    model_id: str,
    **kwargs: Any,
) -> TokenizerBase:
    """Get a tokenizer by name."""
    is_from_mistral_org = "mistral" in str(model_id)
    if is_from_mistral_org and tokenizer_mode != "mistral":
        warnings.warn(
            "It is strongly recommended to run mistral models with "
            'tokenizer-mode "mistral"` to ensure correct '
            "encoding and decoding.",
            FutureWarning,
            stacklevel=2,
        )

    from private_gpt.components.llm.tokenizers.registry import TokenizerRegistry

    base: TokenizerBase = TokenizerRegistry.get_tokenizer(
        tokenizer_mode=tokenizer_mode, model_id=model_id, **kwargs
    )
    return base


def get_prompt_style(
    prompt_style: str,
    *args: Any,
    **kwargs: Any,
) -> PromptStyleBase:
    """Get a tokenizer formatter by name."""
    from private_gpt.components.llm.prompt_styles.registry import PromptStyleRegistry

    return PromptStyleRegistry.get_prompt_style(
        prompt_style,
        *args,
        **kwargs,
    )
