from pathlib import Path
from typing import Any, TypeVar, cast

from transformers import (  # type: ignore[import-not-found]  # ty:ignore[unresolved-import]
    PreTrainedTokenizerBase,
    ProcessorMixin,
)

from private_gpt.components.llm.tokenizers.tokenizer_base import (
    AudioLike,
    ImageLike,
    TextLike,
    TokenizedInput,
    TokenizerBase,
)
from private_gpt.components.llm.tokenizers.utils import build_minimal_messages
from private_gpt.utils.dependencies import format_missing_dependency_message

T = TypeVar("T")


class HuggingFaceTokenizer(TokenizerBase):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        is_multimodal: bool = False,
        processor: ProcessorMixin | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._is_multimodal = is_multimodal
        self._processor = processor

        if self._processor:
            minimal_conversation = build_minimal_messages()
            self._empty_conversation: Any = self._tokenizer.apply_chat_template(
                minimal_conversation,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | Path,
        local_files_only: bool = False,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> "HuggingFaceTokenizer":
        """Load tokenizer from pretrained model with intelligent caching.

        If the model is already cached locally, it will automatically use
        offline mode to avoid network calls.

        Args:
            model_id: Model identifier or local path
            local_files_only: Force offline mode (no downloads)
            cache_dir: Custom cache directory
            force_download: Force re-download even if cached
            trust_remote_code: Allow custom code from model repositories
            **kwargs: Additional arguments for AutoProcessor
        """
        try:
            from transformers import AutoProcessor  # ty:ignore[unresolved-import]
        except ImportError as e:
            raise ImportError(
                format_missing_dependency_message(
                    "Transformers",
                )
            ) from e

        try:
            is_multimodal = False
            processor = None
            loaded: Any = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path=model_id,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
                force_download=force_download,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

            # Extract tokenizer from multimodal processor if needed
            tokenizer: PreTrainedTokenizerBase
            if hasattr(loaded, "tokenizer"):
                processor = cast(ProcessorMixin, loaded)
                tokenizer = cast(PreTrainedTokenizerBase, loaded.tokenizer)
                is_multimodal = True
            else:
                tokenizer = cast(PreTrainedTokenizerBase, loaded)

            return cls(tokenizer, is_multimodal=is_multimodal, processor=processor)

        except OSError as e:
            if local_files_only:
                raise FileNotFoundError(
                    f"Local model files not found at '{model_id}'. "
                    f"Ensure the model is downloaded locally."
                ) from e
            raise ValueError(f"Could not load tokenizer from '{model_id}': {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer: {e}") from e

    @classmethod
    def is_available(cls, model_id: str | Path | None, **kwargs: Any) -> bool:
        return bool(model_id)

    @property
    def all_special_tokens(self) -> list[str]:
        tokens: list[str] = self._tokenizer.all_special_tokens
        return tokens

    @property
    def all_special_ids(self) -> list[int]:
        ids: list[int] = self._tokenizer.all_special_ids
        return ids

    @property
    def bos_token_id(self) -> int:
        return cast(int, self._tokenizer.bos_token_id)

    @property
    def eos_token_id(self) -> int:
        return cast(int, self._tokenizer.eos_token_id)

    @property
    def is_fast(self) -> bool:
        if hasattr(self._tokenizer, "is_fast"):
            return bool(self._tokenizer.is_fast)
        raise NotImplementedError()

    @property
    def vocab_size(self) -> int:
        if hasattr(self._tokenizer, "vocab_size"):
            return int(self._tokenizer.vocab_size)
        raise NotImplementedError()

    @property
    def max_token_id(self) -> int:
        if hasattr(self._tokenizer, "max_token_id"):
            return int(self._tokenizer.max_token_id)
        raise NotImplementedError()

    @property
    def is_multimodal(self) -> bool:
        return self._is_multimodal

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
        """Tokenize text, images, audio, and/or video."""
        text_input_ids: list[int] = []
        mm_input_ids: list[int] = []

        if texts:
            text_input_ids = self.calculate_text_input_ids(texts=texts)

        if images or audios:
            mm_input_ids = self.calculate_mm_input_ids(
                texts=texts,
                images=images,
                audios=audios,
            )

        return TokenizedInput(
            input_ids=text_input_ids + mm_input_ids,
        )

    def calculate_text_input_ids(
        self,
        texts: TextLike | None = None,
    ) -> list[int]:
        """Estimate tokens for text using tokenizer."""
        if not texts:
            return []

        batch_encoding = self._tokenizer(
            texts,
            add_special_tokens=False,
        )
        text_input_ids: list[int] = batch_encoding["input_ids"]
        return text_input_ids

    def calculate_mm_input_ids(
        self,
        texts: TextLike | None = None,
        images: ImageLike | None = None,
        audios: AudioLike | None = None,
    ) -> list[int]:
        """Estimate tokens for images and audio using processor."""
        if not self._processor:
            raise ValueError(
                "Multimodal input provided but tokenizer is not multimodal"
            )

        current_conversation: Any = self._tokenizer.apply_chat_template(
            build_minimal_messages(images=images, audios=audios),
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        total_input_ids = current_conversation["input_ids"][0]
        baseline_input_ids = self._empty_conversation["input_ids"][0]
        return [int(id) for id in total_input_ids if id not in baseline_input_ids]

    def get_vocab(self) -> dict[str, int]:
        vocab: dict[str, int] = self._tokenizer.get_vocab()
        return vocab

    def get_added_vocab(self) -> dict[str, int]:
        raise NotImplementedError()

    def encode(self, text: str, add_special_tokens: bool | None = None) -> list[int]:
        encoded: list[int] = self._tokenizer.encode(
            text, add_special_tokens=add_special_tokens or True
        )
        return encoded

    def support_chat_template(self, tokenizer: Any) -> bool:
        return tokenizer and hasattr(tokenizer, "apply_chat_template")

    def apply_chat_template(
        self,
        conversation: list[dict[str, str | list[dict[str, str]]]],
        tools: list[dict[str, Any]] | None = None,
        documents: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> list[int] | str:
        result: list[int] | str = self._tokenizer.apply_chat_template(
            conversation, tools=tools, documents=documents, **kwargs
        )
        return result

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        converted: str = self._tokenizer.convert_tokens_to_string(tokens)
        return converted

    def decode(self, ids: list[int] | int, skip_special_tokens: bool = True) -> str:
        decoded: str = self._tokenizer.decode(
            ids, skip_special_tokens=skip_special_tokens
        )
        return decoded

    def convert_ids_to_tokens(
        self, ids: list[int], skip_special_tokens: bool = True
    ) -> list[str]:
        raise NotImplementedError()
