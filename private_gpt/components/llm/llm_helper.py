from collections.abc import Callable
from typing import Any

from llama_index.core.llms import LLM
from llama_index.core.multi_modal_llms import MultiModalLLMMetadata

from private_gpt.components.llm.tokenizers.tokenizer_base import (
    AudioLike,
    ImageLike,
    TextLike,
    TokenizedInput,
    TokenizerBase,
)
from private_gpt.di import get_global_injector
from private_gpt.settings.settings import LLMModelConfig


def is_multimodal(
    llm: LLM,
) -> bool:
    return isinstance(llm.metadata, MultiModalLLMMetadata)


def supports_images(llm: LLM, config: LLMModelConfig) -> bool:
    return is_multimodal(llm) and bool(config.support_image)


def max_images_supported(
    llm: LLM,
    config: LLMModelConfig,
) -> int:
    return int(config.support_image or 0) if supports_images(llm, config) else 0


def supports_audio(
    llm: LLM,
    config: LLMModelConfig,
) -> bool:
    return is_multimodal(llm) and bool(config.support_audio)


def max_audios_supported(
    llm: LLM,
    config: LLMModelConfig,
) -> int:
    return int(config.support_audio or 0) if supports_audio(llm, config) else 0


TokenizerFn = Callable[..., TokenizedInput]


def get_tokenizer_fn(tokenizer: TokenizerBase | None) -> TokenizerFn | None:
    if tokenizer is None:
        return None

    def fn(
        texts: TextLike | None = None,
        images: ImageLike | None = None,
        audios: AudioLike | None = None,
        **kwargs: Any,
    ) -> TokenizedInput:
        tokens = tokenizer(texts=texts, images=images, audios=audios, **kwargs)
        return tokens

    return fn


def get_tokenizer() -> TokenizerFn | None:
    from private_gpt.components.llm.llm_component import LLMComponent

    llm_component = get_global_injector().get(LLMComponent)
    default_tokenizer = llm_component.tokenizer
    return get_tokenizer_fn(default_tokenizer)
