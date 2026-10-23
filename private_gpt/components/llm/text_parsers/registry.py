from collections.abc import Callable
from typing import Any

from private_gpt.components.llm.text_parsers.text_parser_base import TextParserBase

TextParserProvider = type[TextParserBase] | Callable[..., TextParserBase | None]

_EXTERNAL_TEXT_PARSER_FACTORIES: dict[str, TextParserProvider] = {}


def register_text_parser_factory(
    prompt_style: str,
    factory: TextParserProvider,
) -> None:
    _EXTERNAL_TEXT_PARSER_FACTORIES[prompt_style] = factory


def _build_openai_text_parser(*args: Any, **kwargs: Any) -> TextParserBase | None:
    from private_gpt.components.llm.text_parsers.harmony_text_parser import (
        HarmonyTextParser,
    )

    return HarmonyTextParser(*args, **kwargs)


_BUILTIN_TEXT_PARSER_FACTORIES: dict[str, TextParserProvider] = {
    "openai": _build_openai_text_parser,
}


class TextParserRegistry:
    @staticmethod
    def get_text_parser(
        prompt_style: str,
        *args: Any,
        **kwargs: Any,
    ) -> TextParserBase | None:
        factory = _EXTERNAL_TEXT_PARSER_FACTORIES.get(
            prompt_style
        ) or _BUILTIN_TEXT_PARSER_FACTORIES.get(prompt_style)
        if factory is not None:
            return factory(*args, **kwargs)
        return TextParserBase(*args, **kwargs)
