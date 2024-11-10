import logging
from collections.abc import Callable
from typing import Any

from injector import inject, singleton
from llama_index.core.llms import LLM, MockLLM
from llama_index.core.settings import Settings as LlamaIndexSettings
from llama_index.core.utils import set_global_tokenizer
from transformers import AutoTokenizer  # type: ignore

from private_gpt.components.llm.prompt_helper import get_prompt_style
from private_gpt.paths import models_cache_path, models_path
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class LLMComponent:
    llm: LLM

    @inject
    def __init__(self, settings: Settings) -> None:
        llm_mode = settings.llm.mode
        if settings.llm.tokenizer and settings.llm.mode != "mock":
            # Try to download the tokenizer. If it fails, the LLM will still work
            # using the default one, which is less accurate.
            try:
                set_global_tokenizer(
                    AutoTokenizer.from_pretrained(
                        pretrained_model_name_or_path=settings.llm.tokenizer,
                        cache_dir=str(models_cache_path),
                        token=settings.huggingface.access_token,
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Failed to download tokenizer {settings.llm.tokenizer}: {e!s}"
                    f"Please follow the instructions in the documentation to download it if needed: "
                    f"https://docs.privategpt.dev/installation/getting-started/troubleshooting#tokenizer-setup."
                    f"Falling back to default tokenizer."
                )

        logger.info("Initializing the LLM in mode=%s", llm_mode)
        match settings.llm.mode:
            case "llamacpp":
                try:
                    from llama_index.llms.llama_cpp import LlamaCPP  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "Local dependencies not found, install with `poetry install --extras llms-llama-cpp`"
                    ) from e

                prompt_style = get_prompt_style(settings.llm.prompt_style)
                settings_kwargs = {
                    "tfs_z": settings.llamacpp.tfs_z,  # ollama and llama-cpp
                    "top_k": settings.llamacpp.top_k,  # ollama and llama-cpp
                    "top_p": settings.llamacpp.top_p,  # ollama and llama-cpp
                    "repeat_penalty": settings.llamacpp.repeat_penalty,  # ollama llama-cpp
                    "n_gpu_layers": -1,
                    "offload_kqv": True,
                }
                self.llm = LlamaCPP(
                    model_path=str(models_path / settings.llamacpp.llm_hf_model_file),
                    temperature=settings.llm.temperature,
                    max_new_tokens=settings.llm.max_new_tokens,
                    context_window=settings.llm.context_window,
                    generate_kwargs={},
                    callback_manager=LlamaIndexSettings.callback_manager,
                    # All to GPU
                    model_kwargs=settings_kwargs,
                    # transform inputs into Llama2 format
                    messages_to_prompt=prompt_style.messages_to_prompt,
                    completion_to_prompt=prompt_style.completion_to_prompt,
                    verbose=True,
                )

            case "sagemaker":
                try:
                    from private_gpt.components.llm.custom.sagemaker import SagemakerLLM
                except ImportError as e:
                    raise ImportError(
                        "Sagemaker dependencies not found, install with `poetry install --extras llms-sagemaker`"
                    ) from e

                self.llm = SagemakerLLM(
                    endpoint_name=settings.sagemaker.llm_endpoint_name,
                    max_new_tokens=settings.llm.max_new_tokens,
                    context_window=settings.llm.context_window,
                )
            case "openai":
                try:
                    from llama_index.llms.openai import OpenAI  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "OpenAI dependencies not found, install with `poetry install --extras llms-openai`"
                    ) from e

                openai_settings = settings.openai
                self.llm = OpenAI(
                    api_base=openai_settings.api_base,
                    api_key=openai_settings.api_key,
                    model=openai_settings.model,
                )
            case "openailike":
                try:
                    from llama_index.llms.openai_like import OpenAILike  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "OpenAILike dependencies not found, install with `poetry install --extras llms-openai-like`"
                    ) from e
                prompt_style = get_prompt_style(settings.llm.prompt_style)
                openai_settings = settings.openai
                self.llm = OpenAILike(
                    api_base=openai_settings.api_base,
                    api_key=openai_settings.api_key,
                    model=openai_settings.model,
                    is_chat_model=True,
                    max_tokens=settings.llm.max_new_tokens,
                    api_version="",
                    temperature=settings.llm.temperature,
                    context_window=settings.llm.context_window,
                    messages_to_prompt=prompt_style.messages_to_prompt,
                    completion_to_prompt=prompt_style.completion_to_prompt,
                    tokenizer=settings.llm.tokenizer,
                    timeout=openai_settings.request_timeout,
                    reuse_client=False,
                )
            case "ollama":
                try:
                    from llama_index.llms.ollama import Ollama  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "Ollama dependencies not found, install with `poetry install --extras llms-ollama`"
                    ) from e

                ollama_settings = settings.ollama

                settings_kwargs = {
                    "tfs_z": ollama_settings.tfs_z,  # ollama and llama-cpp
                    "num_predict": ollama_settings.num_predict,  # ollama only
                    "top_k": ollama_settings.top_k,  # ollama and llama-cpp
                    "top_p": ollama_settings.top_p,  # ollama and llama-cpp
                    "repeat_last_n": ollama_settings.repeat_last_n,  # ollama
                    "repeat_penalty": ollama_settings.repeat_penalty,  # ollama llama-cpp
                }

                # calculate llm model. If not provided tag, it will be use latest
                model_name = (
                    ollama_settings.llm_model + ":latest"
                    if ":" not in ollama_settings.llm_model
                    else ollama_settings.llm_model
                )

                llm = Ollama(
                    model=model_name,
                    base_url=ollama_settings.api_base,
                    temperature=settings.llm.temperature,
                    context_window=settings.llm.context_window,
                    additional_kwargs=settings_kwargs,
                    request_timeout=ollama_settings.request_timeout,
                )

                if ollama_settings.autopull_models:
                    from private_gpt.utils.ollama import check_connection, pull_model

                    if not check_connection(llm.client):
                        raise ValueError(
                            f"Failed to connect to Ollama, "
                            f"check if Ollama server is running on {ollama_settings.api_base}"
                        )
                    pull_model(llm.client, model_name)

                if (
                    ollama_settings.keep_alive
                    != ollama_settings.model_fields["keep_alive"].default
                ):
                    # Modify Ollama methods to use the "keep_alive" field.
                    def add_keep_alive(func: Callable[..., Any]) -> Callable[..., Any]:
                        def wrapper(*args: Any, **kwargs: Any) -> Any:
                            kwargs["keep_alive"] = ollama_settings.keep_alive
                            return func(*args, **kwargs)

                        return wrapper

                    Ollama.chat = add_keep_alive(Ollama.chat)  # type: ignore
                    Ollama.stream_chat = add_keep_alive(Ollama.stream_chat)  # type: ignore
                    Ollama.complete = add_keep_alive(Ollama.complete)  # type: ignore
                    Ollama.stream_complete = add_keep_alive(Ollama.stream_complete)  # type: ignore

                self.llm = llm

            case "azopenai":
                try:
                    from llama_index.llms.azure_openai import (  # type: ignore
                        AzureOpenAI,
                    )
                except ImportError as e:
                    raise ImportError(
                        "Azure OpenAI dependencies not found, install with `poetry install --extras llms-azopenai`"
                    ) from e

                azopenai_settings = settings.azopenai
                self.llm = AzureOpenAI(
                    model=azopenai_settings.llm_model,
                    deployment_name=azopenai_settings.llm_deployment_name,
                    api_key=azopenai_settings.api_key,
                    azure_endpoint=azopenai_settings.azure_endpoint,
                    api_version=azopenai_settings.api_version,
                )
            case "gemini":
                try:
                    from llama_index.llms.gemini import (  # type: ignore
                        Gemini,
                    )
                except ImportError as e:
                    raise ImportError(
                        "Google Gemini dependencies not found, install with `poetry install --extras llms-gemini`"
                    ) from e
                gemini_settings = settings.gemini
                self.llm = Gemini(
                    model_name=gemini_settings.model, api_key=gemini_settings.api_key
                )
            case "mock":
                self.llm = MockLLM()
