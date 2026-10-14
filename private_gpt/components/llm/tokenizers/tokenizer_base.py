import asyncio
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

# Concurrent requests used by the default ``aencode_batch`` for backends that only
# tokenize one text per request.
DEFAULT_TOKENIZE_CONCURRENCY = 32
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

    def encode_batch(
        self, texts: Sequence[str], add_special_tokens: bool | None = None
    ) -> list[list[int]]:
        """Tokenize several texts, in order.

        The default calls ``encode`` once per text. Backends that can tokenize a
        batch in one request should override it, like embedding models do with
        ``_get_text_embeddings``.
        """
        return [self.encode(text, add_special_tokens) for text in texts]

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

    async def aencode_batch(
        self,
        texts: Sequence[str],
        add_special_tokens: bool | None = None,
        concurrency: int = DEFAULT_TOKENIZE_CONCURRENCY,
    ) -> list[list[int]]:
        """Tokenize several texts, in order.

        The default sends up to ``concurrency`` ``aencode`` requests at a time.
        Backends with a native batch request should override it.
        """
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def one(text: str) -> list[int]:
            async with semaphore:
                return await self.aencode(text, add_special_tokens)

        return list(await asyncio.gather(*(one(text) for text in texts)))

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
