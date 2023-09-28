from injector import singleton
from llama_index.llms import CompletionResponseGen, MockLLM

from private_gpt.llm.sagemaker import SagemakerLLM
from private_gpt.util.lazy_dict import LazyDict


@singleton
class CompletionsService:
    def __init__(self) -> None:
        self._models = LazyDict(
            {
                "mock": lambda: MockLLM(),
                "llama-sagemaker": lambda: SagemakerLLM(
                    endpoint_name="huggingface-pytorch-tgi-inference-2023-09-25-19-53-32-140"
                ),
            }
        )

    def stream_complete(
        self, prompt: str, *, model: str | None = None
    ) -> CompletionResponseGen:
        if model is None:
            model = "llama-sagemaker"

        return self._models[model].stream_complete(prompt)  # type: ignore
