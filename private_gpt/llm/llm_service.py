import asyncio
from collections.abc import AsyncGenerator, Sequence

from injector import inject, singleton
from llama_index.llms import MockLLM
from llama_index.llms.base import (
    LLM,
    ChatMessage,
    ChatResponseGen,
    CompletionResponseGen,
    ChatResponse,
)
from llama_index.llms.llama_utils import completion_to_prompt, messages_to_prompt
from llama_index.vector_stores.types import VectorStore

from private_gpt.constants import MODELS_PATH
from private_gpt.settings import settings
from private_gpt.typing import T


async def _yielding_if_cpu_bound(
    source_generator: AsyncGenerator[T, None]
) -> AsyncGenerator[T, None]:
    """Yield the main loop for CPU bound tasks, that is, the model is running locally.

    When running models locally the main loop (like fastapi) will block
    if we don't yield control at some point making it "buffer" the whole stream.
    """
    cpu_bound = settings.llm.mode == "local"
    async for item in source_generator:
        yield item
        if cpu_bound:
            await asyncio.sleep(0)


@singleton
class LLMService:
    llm: LLM
    vector_store: VectorStore

    @inject
    def __init__(self) -> None:
        match settings.llm.mode:
            case "local":
                from llama_index.llms import LlamaCPP

                self.llm = LlamaCPP(
                    model_path=str(MODELS_PATH / settings.local.model_file),
                    temperature=0.1,
                    # llama2 has a context window of 4096 tokens,
                    # but we set it lower to allow for some wiggle room
                    context_window=3900,
                    generate_kwargs={},
                    # set to at least 1 to use GPU
                    model_kwargs={"n_gpu_layers": 1},
                    # transform inputs into Llama2 format
                    messages_to_prompt=messages_to_prompt,
                    completion_to_prompt=completion_to_prompt,
                    verbose=True,
                )

            case "sagemaker":
                from private_gpt.llm.custom.sagemaker import SagemakerLLM

                self.llm = SagemakerLLM(
                    endpoint_name=settings.sagemaker.endpoint_name,
                )
            case "openai":
                from llama_index.llms import OpenAI

                openai_settings = settings.openai.api_key
                self.llm = OpenAI(api_key=openai_settings)
            case "mock":
                self.llm = MockLLM()

    def stream_complete(self, message: str) -> CompletionResponseGen:
        stream = self.llm.stream_complete(message)
        return stream

    def stream_chat(self, messages: Sequence[ChatMessage]) -> ChatResponseGen:
        stream = self.llm.stream_chat(messages)
        return stream

    def chat(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        response = self.llm.chat(messages)
        return response
