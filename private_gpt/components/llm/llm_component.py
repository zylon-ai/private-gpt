import logging

from injector import inject, singleton
from llama_index.llms import MockLLM
from llama_index.llms.base import LLM
from llama_index.llms.llama_utils import completion_to_prompt, messages_to_prompt
from llama_index.llms import LlamaCPP

from private_gpt.paths import models_path
from private_gpt.settings.settings import Settings


logger = logging.getLogger(__name__)




# Inside LLMComponent class

@singleton
class LLMComponent:
    llm: LLM

    @inject
    def __init__(self, settings: Settings) -> None:
        llm_mode = settings.chatbot.llm_mode  # Move this line outside the if block
        chatbot_mode = settings.chatbot.mode

        if chatbot_mode == "hybrid":
            llm_mode = settings.chatbot.llm_mode
            #embedding_mode = settings.chatbot.embedding_mode

        match llm_mode:
            case "local":
                self.llm = self._create_local_llm(settings)
            case "sagemaker":
                self.llm = self._create_sagemaker_llm(settings)
            case "openai":
                self.llm = self._create_openai_llm(settings)
            case "mock":
                self.llm = MockLLM()
            case _:
                raise ValueError(f"Unsupported llm mode: {llm_mode}")

    def _create_local_llm(self, settings: Settings) -> LLM:
        return LlamaCPP(
            model_path=str(models_path / settings.local.llm_hf_model_file),
            temperature=0.1,
            context_window=3900,
            generate_kwargs={},
            model_kwargs={"n_gpu_layers": -1},
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            verbose=True,
        )

    def _create_sagemaker_llm(self, settings: Settings) -> LLM:
        from private_gpt.components.llm.custom.sagemaker import SagemakerLLM
        return SagemakerLLM(endpoint_name=settings.sagemaker.llm_endpoint_name)

    def _create_openai_llm(self, settings: Settings) -> LLM:
        from llama_index.llms import OpenAI
        openai_settings = settings.openai.api_key
        return OpenAI(api_key=openai_settings)

