from injector import inject, singleton
from llama_index.llms import MockLLM
from llama_index.llms.base import LLM

from private_gpt.components.llm.prompt_helper import get_prompt_style
from private_gpt.paths import models_path
from private_gpt.settings.settings import Settings


@singleton
class LLMComponent:
    llm: LLM

    @inject
    def __init__(self, settings: Settings) -> None:
        match settings.llm.mode:
            case "local":
                from llama_index.llms import LlamaCPP

                prompt_style_cls = get_prompt_style(settings.local.prompt_style)
                prompt_style = prompt_style_cls(
                    default_system_prompt=settings.local.default_system_prompt
                )

                self.llm = LlamaCPP(
                    model_path=str(models_path / settings.local.llm_hf_model_file),
                    temperature=0.1,
                    # llama2 has a context window of 4096 tokens,
                    # but we set it lower to allow for some wiggle room
                    context_window=3900,
                    generate_kwargs={},
                    # All to GPU
                    model_kwargs={"n_gpu_layers": -1},
                    # transform inputs into Llama2 format
                    messages_to_prompt=prompt_style.messages_to_prompt,
                    completion_to_prompt=prompt_style.completion_to_prompt,
                    verbose=True,
                )

            case "sagemaker":
                from private_gpt.components.llm.custom.sagemaker import SagemakerLLM

                self.llm = SagemakerLLM(
                    endpoint_name=settings.sagemaker.llm_endpoint_name,
                )
            case "openai":
                from llama_index.llms import OpenAI

                openai_settings = settings.openai.api_key
                self.llm = OpenAI(api_key=openai_settings)
            case "mock":
                self.llm = MockLLM()
