from collections.abc import Sequence
from typing import Any

from private_gpt.components.llm.tokenizers.tokenizer_base import (
    AudioLike,
    ImageLike,
    TextLike,
    TokenizedInput,
    TokenizerBase,
)
from private_gpt.utils.dependencies import format_missing_dependency_message

_DEFAULT_TIKTOKEN_ENCODING = "cl100k_base"


class TikTokenTokenizer(TokenizerBase):
    """Tokenizer backed by tiktoken for local token counting."""

    def __init__(self, encoding: Any, encoding_name: str) -> None:
        self._encoding = encoding
        self._encoding_name = encoding_name

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        **kwargs: Any,
    ) -> "TikTokenTokenizer":
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError(format_missing_dependency_message("tiktoken")) from e

        # An explicit encoding_name (e.g. "cl100k_base") takes precedence over
        # auto-detection so callers using OpenAI-compatible/local model names
        # can always specify the exact encoding they need.
        explicit_encoding_name: str | None = kwargs.get("encoding_name")
        if explicit_encoding_name:
            encoding = tiktoken.get_encoding(explicit_encoding_name)
            return cls(encoding=encoding, encoding_name=explicit_encoding_name)

        if model_id:
            # 1. Try the tiktoken model registry (covers all known OpenAI model ids).
            try:
                encoding = tiktoken.encoding_for_model(model_id)
                return cls(encoding=encoding, encoding_name=encoding.name)
            except (KeyError, ValueError):
                pass

            # 2. model_id might itself be an encoding name (e.g. "cl100k_base").
            try:
                encoding = tiktoken.get_encoding(model_id)
                return cls(encoding=encoding, encoding_name=model_id)
            except (KeyError, ValueError):
                pass

        # 3. Unknown model id — fall back to the default encoding with a warning.
        encoding = tiktoken.get_encoding(_DEFAULT_TIKTOKEN_ENCODING)
        return cls(encoding=encoding, encoding_name=_DEFAULT_TIKTOKEN_ENCODING)

    @property
    def all_special_tokens(self) -> list[str]:
        return []

    @property
    def all_special_ids(self) -> list[int]:
        return []

    @property
    def bos_token_id(self) -> int:
        raise NotImplementedError("TikTokenTokenizer does not expose token ids")

    @property
    def eos_token_id(self) -> int:
        raise NotImplementedError("TikTokenTokenizer does not expose token ids")

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        n_vocab = getattr(self._encoding, "n_vocab", None)
        return int(n_vocab) if n_vocab is not None else 0

    @property
    def max_token_id(self) -> int:
        return max(self.vocab_size - 1, 0)

    @property
    def is_multimodal(self) -> bool:
        return False

    def __call__(
        self,
        texts: TextLike | None = None,
        images: ImageLike | None = None,
        audios: AudioLike | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> TokenizedInput:
        del add_special_tokens, truncation, max_length, kwargs

        if images or audios:
            raise NotImplementedError(
                "TikTokenTokenizer only supports text token counting"
            )
        if texts is None:
            return TokenizedInput(input_ids=[])

        if isinstance(texts, str):
            return TokenizedInput(input_ids=self._encoding.encode(texts))

        if isinstance(texts, Sequence):
            input_ids: list[int] = []
            for text in texts:
                input_ids.extend(self._encoding.encode(str(text)))
            return TokenizedInput(input_ids=input_ids)

        return TokenizedInput(input_ids=self._encoding.encode(str(texts)))

    def get_vocab(self) -> dict[str, int]:
        raise NotImplementedError(
            "TikTokenTokenizer does not expose a local vocabulary"
        )

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    def encode(self, text: str, add_special_tokens: bool | None = None) -> list[int]:
        del add_special_tokens
        return list(self._encoding.encode(text))

    def support_chat_template(self, tokenizer: Any) -> bool:
        return False

    def apply_chat_template(
        self,
        conversation: list[dict[str, str | list[dict[str, str]]]],
        tools: list[dict[str, Any]] | None = None,
        documents: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> list[int] | str:
        del conversation, tools, documents, kwargs
        raise NotImplementedError(
            "TikTokenTokenizer cannot render chat templates locally"
        )

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def decode(self, ids: list[int] | int, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        if isinstance(ids, int):
            ids = [ids]
        return str(self._encoding.decode(ids))

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        del skip_special_tokens
        return [self._encoding.decode([token_id]) for token_id in ids]
