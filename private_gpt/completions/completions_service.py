import asyncio
from collections.abc import AsyncGenerator, Sequence

from injector import inject, singleton
from llama_index.llms import MockLLM, OpenAI
from llama_index.llms.base import (
    LLM,
    ChatMessage,
    ChatResponseAsyncGen,
    CompletionResponseAsyncGen,
)

from private_gpt.constants import MODELS_PATH
from private_gpt.settings import settings
from private_gpt.typing import T


async def _yielding_if_cpu_bound(
    model_name: str, source_generator: AsyncGenerator[T, None]
) -> AsyncGenerator[T, None]:
    """Yield the main loop for CPU bound tasks, that is, the model is running locally.

    When running models locally the main loop (like fastapi) will block
    if we don't yield control at some point making it "buffer" the whole stream.
    """
    cpu_bound = model_name in ("local_llm", "mock")
    async for item in source_generator:
        yield item
        if cpu_bound:
            await asyncio.sleep(0)


@singleton
class CompletionsService:
    _models: dict[str, LLM]

    @inject
    def __init__(self) -> None:
        self._models = {
            "mock": MockLLM(),
        }

        if settings.local_llm.enabled:
            from llama_index.llms import LlamaCPP
            from llama_index.llms.llama_utils import (
                completion_to_prompt,
                messages_to_prompt,
            )

            self._models["local_llm"] = LlamaCPP(
                model_path=str(MODELS_PATH / settings.local_llm.model_file),
                temperature=0.1,
                # llama2 has a context window of 4096 tokens,
                # but we set it lower to allow for some wiggle room
                context_window=3900,
                max_new_tokens=500,
                generate_kwargs={},
                # set to at least 1 to use GPU
                model_kwargs={"n_gpu_layers": 1},
                # transform inputs into Llama2 format
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                verbose=True,
            )

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
        self, messages: Sequence[ChatMessage], *, model_name: str | None = None
    ) -> ChatResponseAsyncGen:
        if model_name is None:
            model_name = settings.llm.default_llm
        model = self._models[model_name]
        stream_task = await asyncio.to_thread(model.astream_chat, messages)
        return _yielding_if_cpu_bound(model_name, await stream_task)
