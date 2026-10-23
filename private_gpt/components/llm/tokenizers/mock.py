from typing import Any

from llama_index.core import get_tokenizer

from private_gpt.components.llm.tokenizers.tokenizer_base import (
    AudioLike,
    ImageLike,
    TextLike,
    TokenizedInput,
    TokenizerBase,
)


class MockTokenizer(TokenizerBase):
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "TokenizerBase":
        return cls()

    @property
    def all_special_tokens_extended(self) -> list[str]:
        return []

    @property
    def all_special_tokens(self) -> list[str]:
        return []

    @property
    def all_special_ids(self) -> list[int]:
        return []

    @property
    def bos_token_id(self) -> int:
        return 0

    @property
    def eos_token_id(self) -> int:
        return 0

    @property
    def sep_token(self) -> str:
        return ""

    @property
    def pad_token(self) -> str:
        return ""

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        return 0

    @property
    def max_token_id(self) -> int:
        return 0

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
        tokenizer = get_tokenizer()
        input_ids: Any = tokenizer(texts or "")  # type: ignore
        return TokenizedInput(
            input_ids=input_ids,
        )

    def get_vocab(self) -> dict[str, int]:
        return {}

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    def encode_one(
        self, text: str, truncation: bool = False, max_length: int | None = None
    ) -> list[int]:
        return self(text, truncation=truncation, max_length=max_length)

    def encode(self, text: str, add_special_tokens: bool | None = None) -> list[int]:
        return self(text, add_special_tokens=add_special_tokens or False)

    def support_chat_template(self, tokenizer: Any) -> bool:
        return True

    def apply_chat_template(
        self,
        conversation: list[dict[str, str | list[dict[str, str]]]],
        tools: list[dict[str, Any]] | None = None,
        documents: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> list[int] | str:
        return ""

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def decode(self, ids: list[int] | int, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, int):
            ids = [ids]
        return " ".join([str(i) for i in ids])

    def convert_ids_to_tokens(
        self, ids: list[int], skip_special_tokens: bool = True
    ) -> list[str]:
        return [str(i) for i in ids]
