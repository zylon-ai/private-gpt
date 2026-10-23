import asyncio
import concurrent.futures
from collections.abc import Awaitable, Callable
from typing import Any, cast

from llama_index.core.llms import LLM
from llama_index.core.multi_modal_llms import MultiModalLLMMetadata

from private_gpt.components.llm.tokenizers.tokenizer_base import (
    AsyncTokenizerBase,
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
AsyncTokenizerFn = Callable[..., Awaitable[TokenizedInput]]


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


def get_async_tokenizer_fn(tokenizer: AsyncTokenizerBase) -> AsyncTokenizerFn:
    async def fn(
        texts: TextLike | None = None,
        images: ImageLike | None = None,
        audios: AudioLike | None = None,
        **kwargs: Any,
    ) -> TokenizedInput:
        return await tokenizer.acall(
            texts=texts, images=images, audios=audios, **kwargs
        )

    return fn


def as_sync_tokenizer_fn(
    tokenizer_fn: TokenizerFn | AsyncTokenizerFn | None,
) -> TokenizerFn | None:
    """Wrap an AsyncTokenizerFn so it can be called synchronously.

    Runs the coroutine in a fresh event loop on a dedicated thread, avoiding
    conflicts with any running loop in the caller's thread.
    """
    if tokenizer_fn is None:
        return None
    if not asyncio.iscoroutinefunction(tokenizer_fn):
        return cast(TokenizerFn, tokenizer_fn)
    _async_fn = tokenizer_fn

    def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, _async_fn(*args, **kwargs)).result()

    return _sync_wrapper


def get_tokenizer() -> TokenizerFn | None:
    from private_gpt.components.llm.llm_component import LLMComponent

    llm_component = get_global_injector().get(LLMComponent)
    default_tokenizer = llm_component.tokenizer
    return get_tokenizer_fn(default_tokenizer)
