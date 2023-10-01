from collections.abc import Sequence

from injector import inject, singleton
from llama_index.llms import CompletionResponseGen, MockLLM, OpenAI
from llama_index.llms.base import LLM, ChatMessage, ChatResponseGen

from private_gpt.constants import MODELS_PATH
from private_gpt.settings import settings


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
                model_path=str(MODELS_PATH / settings.local_llm.model_name),
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

        sagemaker_settings = settings.sagemaker
        if settings.sagemaker.enabled:
            from private_gpt.llm.sagemaker import SagemakerLLM

            self._models["llama-sagemaker"] = SagemakerLLM(
                endpoint_name=sagemaker_settings.endpoint_name,
            )

        if settings.openai.enabled:
            self._models["openai"] = OpenAI(api_key=settings.openai.api_key)

    def stream_complete(
        self, prompt: str, *, model_name: str | None = None
    ) -> CompletionResponseGen:
        if model_name is None:
            model_name = settings.llm.default_llm

        model = self._models[model_name]
        return model.stream_complete(prompt)

    def stream_chat(
        self, messages: Sequence[ChatMessage], *, model_name: str | None = None
    ) -> ChatResponseGen:
        if model_name is None:
            model_name = settings.llm.default_llm
        model = self._models[model_name]
        return model.stream_chat(messages)
