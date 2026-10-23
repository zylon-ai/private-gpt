import logging
from abc import ABC, abstractmethod
from typing import Any

from llama_index.core.llms import LLM
from pydantic import BaseModel, Field

from private_gpt.components.llm.prompt_helper import get_tokenizer
from private_gpt.components.llm.tokenizers.tokenizer_base import TokenizerBase
from private_gpt.paths import models_cache_path
from private_gpt.settings.settings import LLMModelConfig, Settings

logger = logging.getLogger(__name__)


class LLMInstance(BaseModel):
    llm: LLM = Field(..., description="The LLM instance")
    tokenizer: TokenizerBase | None = Field(
        None, description="Optional tokenizer associated with the LLM"
    )
    alias: str | None = Field(None, description="Optional alias for the LLM")

    class Config:
        arbitrary_types_allowed = True


CACHE_TOKENIZER: dict[str, TokenizerBase] = {}


class LLMFactory(ABC):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def create_llm(self, model_config: LLMModelConfig) -> LLMInstance:
        """Create LLM instance and return the instance and config."""
        tokenizer = self.load_tokenizer(model_config, raise_exception=True)
        llm, alias = self._create_llm(model_config, tokenizer=tokenizer)
        return LLMInstance(llm=llm, tokenizer=tokenizer, alias=alias)

    @abstractmethod
    def _create_llm(
        self, model_config: LLMModelConfig, tokenizer: TokenizerBase | None = None
    ) -> tuple[LLM, str | None]:
        """Create LLM instance, to be implemented by subclasses."""
        pass

    def load_tokenizer(
        self, model_config: LLMModelConfig, raise_exception: bool = True
    ) -> TokenizerBase | None:
        """Load tokenizer if specified in model config."""
        try:
            tokenizer_mode = model_config.tokenizer_mode or "default"
            resolved_model_id = model_config.tokenizer or ""

            cache_key = f"{tokenizer_mode}::{resolved_model_id}"
            if cache_key in CACHE_TOKENIZER:
                return CACHE_TOKENIZER[cache_key]

            kwargs: Any = {"model_id": resolved_model_id}
            if tokenizer_mode == "remote":
                kwargs.update(
                    {
                        "api_base": self.settings.openai.api_base,
                        "api_key": self.settings.openai.api_key,
                        "request_timeout": self.settings.openai.request_timeout,
                    }
                )
            else:
                kwargs.update(
                    {
                        "local_files_only": self.settings.server.network.offline_mode
                        or False,
                        "cache_dir": str(models_cache_path),
                        "token": self.settings.huggingface.access_token,
                    }
                )

            tokenizer = get_tokenizer(
                tokenizer_mode=tokenizer_mode,
                **kwargs,
            )
            CACHE_TOKENIZER[cache_key] = tokenizer
            return tokenizer

        except Exception as e:
            logger.warning(
                f"Failed to load tokenizer for model {model_config.name}: {e}"
            )
            if raise_exception:
                raise e

            return None
