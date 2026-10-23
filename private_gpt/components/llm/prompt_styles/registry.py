from collections.abc import Callable
from typing import Any

from private_gpt.components.llm.prompt_styles.prompt_style_base import PromptStyleBase

PromptStyleProvider = type[PromptStyleBase] | Callable[..., PromptStyleBase]

_EXTERNAL_PROMPT_STYLE_FACTORIES: dict[str, PromptStyleProvider] = {}


def register_prompt_style_factory(
    prompt_style: str,
    factory: PromptStyleProvider,
) -> None:
    _EXTERNAL_PROMPT_STYLE_FACTORIES[prompt_style] = factory


def _build_chat_template_prompt_style(*args: Any, **kwargs: Any) -> PromptStyleBase:
    from private_gpt.components.llm.prompt_styles.chat_template_prompt_style import (
        ChatTemplatePromptStyle,
    )

    return ChatTemplatePromptStyle(*args, **kwargs)


_BUILTIN_PROMPT_STYLE_FACTORIES: dict[str, PromptStyleProvider] = {
    "chat": _build_chat_template_prompt_style,
}


class PromptStyleRegistry:
    @staticmethod
    def get_prompt_style(
        prompt_style: str,
        *args: Any,
        **kwargs: Any,
    ) -> PromptStyleBase:
        factory = _EXTERNAL_PROMPT_STYLE_FACTORIES.get(
            prompt_style
        ) or _BUILTIN_PROMPT_STYLE_FACTORIES.get(prompt_style)
        if factory is None:
            raise ValueError(f"Unknown prompt_style='{prompt_style}'")
        return factory(*args, **kwargs)
