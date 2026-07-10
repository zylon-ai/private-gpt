from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from io import IOBase
from typing import Any


@dataclass
class TokenizedInput(list[int]):
    input_ids: list[int]

    def __post_init__(self) -> None:
        super().__init__(self.input_ids)


TextLike = str | Sequence[str]
ImageLike = Sequence[IOBase]
AudioLike = Sequence[IOBase]


class TokenizerBase(ABC):
    @classmethod
    @abstractmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "TokenizerBase":
        raise NotImplementedError()

    @property
    @abstractmethod
    def all_special_tokens(self) -> list[str]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def all_special_ids(self) -> list[int]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def bos_token_id(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def is_fast(self) -> bool:
        raise NotImplementedError()

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def max_token_id(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def is_multimodal(self) -> bool:
        raise NotImplementedError()

    def __len__(self) -> int:
        return self.vocab_size

    @abstractmethod
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
        raise NotImplementedError()

    @abstractmethod
    def get_vocab(self) -> dict[str, int]:
        raise NotImplementedError()

    @abstractmethod
    def get_added_vocab(self) -> dict[str, int]:
        raise NotImplementedError()

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool | None = None) -> list[int]:
        raise NotImplementedError()

    @abstractmethod
    def support_chat_template(self, tokenizer: Any) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def apply_chat_template(
        self,
        conversation: list[dict[str, str | list[dict[str, str]]]],
        tools: list[dict[str, Any]] | None = None,
        documents: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> list[int] | str:
        raise NotImplementedError()

    @abstractmethod
    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        raise NotImplementedError()

    @abstractmethod
    def decode(self, ids: list[int] | int, skip_special_tokens: bool = True) -> str:
        raise NotImplementedError()

    @abstractmethod
    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        raise NotImplementedError()


class AsyncTokenizerBase(TokenizerBase, ABC):
    """TokenizerBase extension for natively-async tokenizer backends."""

    @abstractmethod
    async def acall(
        self,
        texts: TextLike | None = None,
        images: ImageLike | None = None,
        audios: AudioLike | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> "TokenizedInput":
        raise NotImplementedError()

    @abstractmethod
    async def aencode(
        self, text: str, add_special_tokens: bool | None = None
    ) -> list[int]:
        raise NotImplementedError()

    @abstractmethod
    async def adecode(
        self, ids: list[int] | int, skip_special_tokens: bool = True
    ) -> str:
        raise NotImplementedError()

    @abstractmethod
    async def aapply_chat_template(
        self,
        conversation: list[dict[str, str | list[dict[str, str]]]],
        tools: list[dict[str, Any]] | None = None,
        documents: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> list[int] | str:
        raise NotImplementedError()
