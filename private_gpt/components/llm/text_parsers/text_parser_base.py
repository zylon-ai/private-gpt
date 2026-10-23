from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole

if TYPE_CHECKING:
    from private_gpt.components.llm.tokenizers.tokenizer_base import TokenizerBase


class TextParserBase:
    """Abstract text parser class that should not be used directly.

    Provided and methods should be used in derived classes.

    It is used to extract text content from the model output.

    Parameters
    ----------
    special_tokens
        Per-model list of special-token strings to strip from the generated
        text (e.g. ``["<|im_start|>", "<|im_end|>", "<|endoftext|>"]`` for
        Qwen).  When *None* (the default) no stripping is performed.
    """

    def __init__(self, tokenizer: TokenizerBase, **kwargs: Any) -> None:
        self.model_tokenizer = tokenizer

        self._special_tokens: list[str] | None = kwargs.pop("special_tokens", None)
        self._special_token_re = self._build_re(self._special_tokens)

    def _build_re(self, tokens: list[str] | None) -> re.Pattern[str]:
        if not tokens:
            return re.compile(r"(?!)")  # never matches

        # longest-first so multi-character tokens like <|endoftext|> are
        # tried before shorter substrings that could be inside them.
        tokens = sorted(tokens, key=len, reverse=True)
        escaped = [re.escape(t) for t in tokens if t]
        return re.compile("|".join(escaped))

    def _strip_special_tokens(self, text: str) -> str:
        if not text:
            return text
        return self._special_token_re.sub("", text)

    @classmethod
    def from_prototype(
        cls,
        prototype: TextParserBase,
        **kwargs: Any,
    ) -> TextParserBase:
        """Create a new instance of the ToolParser class."""
        return prototype.__class__(
            prototype.model_tokenizer,
            special_tokens=prototype._special_tokens,
        )

    def extract_text_content(
        self,
        model_output: str,
    ) -> str:
        """Extract text content from a complete model-generated string.

        Used for non-streaming responses where we have the entire model response
        available before sending to the client.

        Parameters:
        model_output: str
            The model-generated string to extract text content from.

        Returns:
        str
            The extracted text content.
        """
        return self._strip_special_tokens(model_output)

    def extract_text_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> ChatResponse | None:
        """Extract text content from a streaming delta.

        Strips per-model special tokens from *delta_text* using a compiled
        regex — no tokenizer round-trips.
        """
        if not delta_text:
            return None

        delta = self._strip_special_tokens(delta_text)
        if not delta:
            return None

        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=current_text),
            delta=delta,
            raw=current_text,
        )

    def close(self) -> None:
        """Close the reasoning parser."""
        pass
