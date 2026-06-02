from collections.abc import Sequence
from typing import Any

import httpx

from private_gpt.components.llm.tokenizers.tokenizer_base import (
    AudioLike,
    ImageLike,
    TextLike,
    TokenizedInput,
    TokenizerBase,
)


class RemoteTokenizeTokenizer(TokenizerBase):
    """Tokenizer backed by remote `/tokenize` and `/detokenize` endpoints.

    Supports the current wire formats exposed by:
    - vLLM: `POST /tokenize` with `{"model": ..., "prompt": ...}`
    - llama.cpp server: `POST /tokenize` with `{"content": ...}`
    """

    TOKENIZE_ENDPOINT = "/tokenize"
    DETOKENIZE_ENDPOINT = "/detokenize"

    def __init__(
        self,
        model_id: str,
        api_base: str,
        api_key: str | None = None,
        request_timeout: float = 120.0,
    ) -> None:
        self.model_id = model_id
        self.api_base = api_base.rstrip("/").rstrip("v1").rstrip("/")
        self.api_key = api_key
        self.request_timeout = request_timeout

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        api_base: str,
        api_key: str | None = None,
        request_timeout: float = 120.0,
        **kwargs: Any,
    ) -> "RemoteTokenizeTokenizer":
        del kwargs
        return cls(
            model_id=model_id,
            api_base=api_base,
            api_key=api_key,
            request_timeout=request_timeout,
        )

    @classmethod
    def is_available(cls, model_id: str, **kwargs: Any) -> bool:
        # TODO: Try to tokenize a random text
        return False

    @property
    def all_special_tokens(self) -> list[str]:
        return []

    @property
    def all_special_ids(self) -> list[int]:
        return []

    @property
    def bos_token_id(self) -> int:
        raise NotImplementedError(
            "RemoteTokenizeTokenizer does not expose BOS token id"
        )

    @property
    def eos_token_id(self) -> int:
        raise NotImplementedError(
            "RemoteTokenizeTokenizer does not expose EOS token id"
        )

    @property
    def is_fast(self) -> bool:
        return False

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
        del add_special_tokens, truncation, max_length, kwargs

        if images or audios:
            raise NotImplementedError(
                "RemoteTokenizeTokenizer only supports text tokenization"
            )
        if texts is None:
            return TokenizedInput(input_ids=[])

        if isinstance(texts, str):
            return TokenizedInput(input_ids=self.encode(texts))

        if isinstance(texts, Sequence):
            input_ids: list[int] = []
            for text in texts:
                input_ids.extend(self.encode(str(text)))
            return TokenizedInput(input_ids=input_ids)

        return TokenizedInput(input_ids=self.encode(str(texts)))

    def get_vocab(self) -> dict[str, int]:
        raise NotImplementedError(
            "RemoteTokenizeTokenizer does not expose a local vocabulary"
        )

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    def encode(self, text: str, add_special_tokens: bool | None = None) -> list[int]:
        del add_special_tokens
        response = self._post_tokenize(text)
        return self._extract_tokens(response)

    def apply_chat_template(
        self,
        conversation: list[dict[str, str | list[dict[str, str]]]],
        tools: list[dict[str, Any]] | None = None,
        documents: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> list[int] | str:
        del conversation, tools, documents, kwargs
        raise NotImplementedError(
            "RemoteTokenizeTokenizer cannot render chat templates locally"
        )

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def decode(self, ids: list[int] | int, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        token_ids = [ids] if isinstance(ids, int) else ids
        response = self._post_detokenize(token_ids)
        return self._extract_text(response)

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        del skip_special_tokens
        raise NotImplementedError(
            "RemoteTokenizeTokenizer does not expose piecewise token strings"
        )

    def _post_tokenize(self, text: str) -> Any:
        tokenize_url = self._build_url(self.TOKENIZE_ENDPOINT)
        last_error: Exception | None = None

        for payload in (
            {"model": self.model_id, "prompt": text},
            {"content": text},
        ):
            try:
                response = httpx.post(
                    tokenize_url,
                    json=payload,
                    headers=self._headers(),
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                last_error = e

        if last_error is None:
            raise ValueError("Remote tokenization failed without an HTTP error")
        raise last_error

    def _post_detokenize(self, token_ids: list[int]) -> Any:
        detokenize_url = self._build_url(self.DETOKENIZE_ENDPOINT)
        last_error: Exception | None = None

        for payload in (
            {"model": self.model_id, "tokens": token_ids},
            {"tokens": token_ids},
        ):
            try:
                response = httpx.post(
                    detokenize_url,
                    json=payload,
                    headers=self._headers(),
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                last_error = e

        if last_error is None:
            raise ValueError("Remote detokenization failed without an HTTP error")
        raise last_error

    def _build_url(self, path: str) -> str:
        return f"{self.api_base}/{path.lstrip('/')}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _extract_tokens(payload: Any) -> list[int]:
        if isinstance(payload, dict):
            tokens = payload.get("tokens")
            if isinstance(tokens, list) and all(
                isinstance(token, int) for token in tokens
            ):
                return tokens
        raise ValueError(
            "Remote tokenizer response did not contain a valid 'tokens' field"
        )

    @staticmethod
    def _extract_text(payload: Any) -> str:
        if isinstance(payload, dict):
            for key in ("content", "text"):
                value = payload.get(key)
                if isinstance(value, str):
                    return value
        raise ValueError(
            "Remote detokenize response did not contain a supported text field"
        )
