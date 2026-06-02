from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole

from private_gpt.components.llm.text_parsers.text_parser_base import TextParserBase
from private_gpt.utils.dependencies import format_missing_dependency_message

if TYPE_CHECKING:
    from private_gpt.components.llm.tokenizers.tokenizer_base import TokenizerBase


def _load_openai_harmony() -> Any:
    try:
        return importlib.import_module("openai_harmony")
    except ImportError as e:
        raise ImportError(
            format_missing_dependency_message(
                "Harmony parser",
                extras=("llm-openai", "llm-openai-compatible"),
            )
        ) from e


def get_encoding() -> Any:
    harmony = _load_openai_harmony()
    return harmony.load_harmony_encoding(harmony.HarmonyEncodingName.HARMONY_GPT_OSS)


logger = logging.getLogger(__name__)


class HarmonyTextParser(TextParserBase):
    def __init__(self, tokenizer: TokenizerBase, **kwargs: Any) -> None:
        super().__init__(tokenizer, **kwargs)
        self._harmony = _load_openai_harmony()
        self._encoding = get_encoding()
        self._streamable_parser = self._harmony.StreamableParser(
            self._encoding, role=self._harmony.Role.ASSISTANT
        )

    def extract_text_content(
        self,
        model_output: str,
    ) -> str:
        parser = self._harmony.StreamableParser(
            self._encoding, role=self._harmony.Role.ASSISTANT
        )
        tokens = self._encoding.encode(model_output, allowed_special="all")

        final_content_parts = []
        for token in tokens:
            parser.process(token)
            if (
                parser.current_channel == "final"
                and parser.current_role == self._harmony.Role.ASSISTANT
                and parser.last_content_delta
            ):
                final_content_parts.append(parser.last_content_delta)

        return "".join(final_content_parts) if final_content_parts else model_output

    def extract_text_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> ChatResponse | None:
        if not delta_text:
            return None

        delta_tokens = self._encoding.encode(delta_text, allowed_special="all")
        for token in delta_tokens:
            self._streamable_parser.process(token)

        if (
            self._streamable_parser.current_channel == "final"
            and self._streamable_parser.current_role == self._harmony.Role.ASSISTANT
            and self._streamable_parser.last_content_delta
        ):
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=current_text,
                ),
                delta=delta_text,
            )

        return None
