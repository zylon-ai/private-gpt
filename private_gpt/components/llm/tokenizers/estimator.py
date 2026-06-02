from typing import Any

from private_gpt.components.llm.tokenizers.tiktoken import TikTokenTokenizer
from private_gpt.components.llm.tokenizers.tokenizer_base import (
    AudioLike,
    ImageLike,
    TextLike,
    TokenizedInput,
    TokenizerBase,
)


class EstimatorTokenizer(TokenizerBase):
    """Tokenizer estimator backed by tiktoken.

    The estimator intentionally mirrors TikTokenTokenizer behavior by delegating all
    token operations to it. The caller should pass the model or encoding name that
    best represents the target tokenizer, e.g. ``gpt-4o``, ``gpt-4``,
    ``o200k_base`` or ``cl100k_base``.
    """

    def __init__(self, tokenizer: TikTokenTokenizer) -> None:
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> "EstimatorTokenizer":
        tokenizer = TikTokenTokenizer.from_pretrained(model_id=model_id, **kwargs)
        return cls(tokenizer)

    @property
    def all_special_tokens(self) -> list[str]:
        return self._tokenizer.all_special_tokens

    @property
    def all_special_ids(self) -> list[int]:
        return self._tokenizer.all_special_ids

    @property
    def bos_token_id(self) -> int:
        return self._tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    @property
    def is_fast(self) -> bool:
        return self._tokenizer.is_fast

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def max_token_id(self) -> int:
        return self._tokenizer.max_token_id

    @property
    def is_multimodal(self) -> bool:
        return self._tokenizer.is_multimodal

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
        return self._tokenizer(
            texts=texts,
            images=images,
            audios=audios,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length,
            **kwargs,
        )

    def get_vocab(self) -> dict[str, int]:
        return self._tokenizer.get_vocab()

    def get_added_vocab(self) -> dict[str, int]:
        return self._tokenizer.get_added_vocab()

    def encode(self, text: str, add_special_tokens: bool | None = None) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def support_chat_template(self, tokenizer: Any) -> bool:
        return self._tokenizer.support_chat_template(tokenizer)

    def apply_chat_template(
        self,
        conversation: list[dict[str, str | list[dict[str, str]]]],
        tools: list[dict[str, Any]] | None = None,
        documents: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> list[int] | str:
        return self._tokenizer.apply_chat_template(
            conversation=conversation,
            tools=tools,
            documents=documents,
            **kwargs,
        )

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self._tokenizer.convert_tokens_to_string(tokens)

    def decode(self, ids: list[int] | int, skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        return self._tokenizer.convert_ids_to_tokens(
            ids,
            skip_special_tokens=skip_special_tokens,
        )
