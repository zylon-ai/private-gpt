import asyncio
from collections.abc import Sequence
from typing import AsyncGenerator

from injector import inject, singleton
from llama_index.llms import MockLLM, OpenAI
from llama_index.llms.base import (
    LLM,
    ChatMessage,
    CompletionResponseAsyncGen,
    ChatResponseAsyncGen,
)

from private_gpt.settings import settings
from private_gpt.typing import T


async def _yielding_if_cpu_bound(
    model_name: str, source_generator: AsyncGenerator[T, None]
) -> AsyncGenerator[T, None]:
    """Yield the main loop for CPU bound tasks, that is, the model is running locally.

    When running models locally the main loop (like fastapi) will block
    if we don't yield control at some point making it "buffer" the whole stream.
    """
    cpu_bound = settings.llm.mode == 'local'
    async for item in source_generator:
        yield item
        if cpu_bound:
            await asyncio.sleep(0)


@singleton
class CompletionsService:
    _models: dict[str, LLM]
    llm: LLM

    @inject
    def __init__(self) -> None:
        self._models = {
            "mock": MockLLM(),
        }
        sagemaker_settings = settings.sagemaker
        if settings.sagemaker.enabled:
            from private_gpt.llm.sagemaker import SagemakerLLM

            self._models["llama-sagemaker"] = SagemakerLLM(
                endpoint_name=sagemaker_settings.endpoint_name,
            )

        if settings.openai.enabled:
            self._models["openai"] = OpenAI(api_key=settings.openai.api_key)

    async def stream_complete(
        self, prompt: str, *, model_name: str | None = None
    ) -> CompletionResponseAsyncGen:
        if model_name is None:
            model_name = settings.llm.default_llm

        model = self._models[model_name]
        stream = await model.astream_complete(prompt)
        return _yielding_if_cpu_bound(model_name, stream)

    async def stream_chat(
        self, message: str, messages: Sequence[ChatMessage]
    ) -> ChatResponseAsyncGen:
        stream_task = await asyncio.to_thread(self.llm.astream_chat, messages)
        return _yielding_if_cpu_bound(await stream_task)
