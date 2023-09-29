from injector import inject, singleton
from llama_index.llms import CompletionResponseGen, CustomLLM, MockLLM

from private_gpt.settings import settings


@singleton
class CompletionsService:
    _models: dict[str, CustomLLM]

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

    def stream_complete(
        self, prompt: str, *, model: str | None = None
    ) -> CompletionResponseGen:
        if model is None:
            model = "llama-sagemaker"

        return self._models[model].stream_complete(prompt)
