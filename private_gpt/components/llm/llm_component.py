import logging
from collections.abc import Callable, Iterator
from typing import Any

from injector import inject, singleton
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.llms import LLM

from private_gpt.components.llm.custom.base import ZylonLLM
from private_gpt.components.llm.discovery import get_models
from private_gpt.components.llm.factories.registry import LLMFactoryRegistry
from private_gpt.components.llm.llm_helper import TokenizerFn, get_tokenizer_fn
from private_gpt.components.llm.registry import LLMInstance, LLMRegistry
from private_gpt.components.llm.tokenizers.tokenizer_base import TokenizerBase
from private_gpt.components.model_discovery.service import are_distinct_api_bases
from private_gpt.settings.settings import LLMModelConfig, Settings

logger = logging.getLogger(__name__)


@singleton
class LLMComponent:
    @inject
    def __init__(self, settings: Settings, llm_registry: LLMRegistry) -> None:
        self.registry = llm_registry
        self.settings = settings

        self.factory_registry = LLMFactoryRegistry(settings)
        self.llm_models = {
            model.name: model
            for model in self.settings.models
            if model.type == "llm" and model.enabled
        }
        self._default_model_id = settings.llm.default_model

        self._initialize_models()

    def _discover_models(self) -> list[LLMModelConfig]:
        if not self.settings.llm.auto_discover_models:
            return []

        return get_models(
            self.settings.openai.api_base,
            self.settings.openai.api_key,
            force_model_kind=are_distinct_api_bases(
                self.settings.openai.embedding_api_key,
                self.settings.openai.embedding_api_base,
            ),
        )

    def _initialize_models(self) -> None:
        for model in self._discover_models():
            if model.name not in self.llm_models:
                self.llm_models[model.name] = model

        if not self.llm_models:
            logger.warning(
                "No LLM models are configured, "
                "or it was impossible to discover any models from the API"
            )

        registered_model_ids: list[str] = []
        for model_id, model_config in list(self.llm_models.items()):
            try:
                logger.info(
                    "Initializing LLM model '%s' with mode='%s'",
                    model_id,
                    model_config.mode,
                )

                factory = self.factory_registry.get_factory(model_config.mode)
                instance = factory.create_llm(model_config)

                alias = instance.alias
                aliases = [alias] if alias and alias != model_id else []
                if model_id == self._default_model_id:
                    aliases.append(LLMRegistry.default())

                registry_instance = LLMInstance(
                    llm=instance.llm, tokenizer=instance.tokenizer
                )
                self.registry.register(model_id, registry_instance, aliases=aliases)
                registered_model_ids.append(model_id)

                logger.info("Successfully registered LLM model '%s'", model_id)

            except Exception as e:
                self.llm_models.pop(model_id, None)
                if model_id == self._default_model_id:
                    raise ValueError(
                        f"Default LLM model '{model_id}' could not be initialized: {e}"
                    ) from e
                logger.warning(
                    "Skipping unavailable LLM model '%s': %s",
                    model_id,
                    e,
                )

        if not self._default_model_id and registered_model_ids:
            self._default_model_id = next(iter(self.llm_models))
            logger.warning(
                "No default LLM model configured. Auto-selecting: '%s'",
                self._default_model_id,
            )

        if self._default_model_id:
            default_instance = self.registry.get(self._default_model_id)
            if not default_instance:
                raise ValueError(
                    f"Default model '{self._default_model_id}' not found in registered models"
                )

            if not self.registry.get(LLMRegistry.default()):
                self.registry.register(LLMRegistry.default(), default_instance)

            logger.info("Set default model to '%s'", self._default_model_id)

    def get_llm(self, model_id: str | None = None) -> LLM:
        target_model = model_id or self._default_model_id

        if not target_model:
            raise ValueError("No model specified and no models are configured")

        llm_instance = self.registry.get(target_model)
        if not llm_instance:
            available = self.registry.get_all_aliases()
            raise ValueError(
                f"Model '{target_model}' not found. Available: {available}"
            )

        return llm_instance.llm

    def get_alias(self, model_id: str | None = None) -> str | None:
        target_model = model_id or self._default_model_id
        if not target_model:
            return None
        aliases = self.registry.get_aliases(target_model)
        return aliases.pop() if aliases else None

    def get_config(self, model_id: str | None = None) -> LLMModelConfig:
        target_model = model_id or self._default_model_id

        if not target_model:
            raise ValueError("No model specified and no models are configured")

        model_config = self.llm_models.get(target_model)
        if not model_config:
            available = list(self.llm_models.keys())
            for alias in self.registry.get_aliases(target_model):
                if alias in self.llm_models:
                    model_config = self.llm_models[alias]
                    return model_config

            raise ValueError(
                f"Model config '{target_model}' not found. Available: {available}"
            )

        return model_config

    def get_tokenizer(self, model_id: str | None = None) -> TokenizerBase | None:
        target_model = model_id or self._default_model_id

        if not target_model:
            raise ValueError("No model specified and no models are configured")

        llm_instance = self.registry.get(target_model)
        if not llm_instance:
            available = self.registry.get_all_aliases()
            raise ValueError(
                f"Model '{target_model}' not found. Available: {available}"
            )

        tokenizer = llm_instance.tokenizer
        return tokenizer

    def filter(
        self, predicate: Callable[[LLM, LLMModelConfig], bool]
    ) -> Iterator[tuple[LLM, LLMModelConfig, TokenizerFn | None]]:
        """Filter LLM components based on a predicate function.

        :param predicate: A function that takes an LLM
         and returns True if it matches the criteria.
        :return: An iterator over the filtered LLM components.
        """
        seen = set()
        for model_id, component in self.registry._registry.items():
            model_config = self.llm_models.get(model_id)
            if (
                model_config
                and id(model_id) not in seen
                and predicate(component.llm, model_config)
            ):
                seen.add(id(model_id))

                tokenizer_fn = (
                    get_tokenizer_fn(component.tokenizer)
                    if component.tokenizer
                    else None
                )
                yield component.llm, model_config, tokenizer_fn

    @property
    def llm(self) -> LLM:
        return self.get_llm()

    @llm.setter
    def llm(self, value: LLM) -> None:
        if not self._default_model_id:
            raise ValueError("No default model configured to set LLM")

        before_tokenizer: TokenizerBase | None = None
        llm_instance = self.registry.get(self._default_model_id)
        if llm_instance:
            before_tokenizer = llm_instance.tokenizer
            self.registry.unregister(self._default_model_id)

        instance = LLMInstance(llm=value, tokenizer=before_tokenizer)
        logger.warning(
            "Directly setting the LLM instance is deprecated. "
            "Use ONLY for testing purposes."
        )
        self.registry.register(self._default_model_id, instance)

    @property
    def alias(self) -> str | None:
        return self.get_alias()

    @property
    def tokenizer(self) -> TokenizerBase | None:
        return self.get_tokenizer()

    def metadata(self, model_id: str | None = None, **kwargs: Any) -> LLMMetadata:
        llm = self.get_llm(model_id)
        llm_metadata: Any = (
            llm.get_metadata(**kwargs) if isinstance(llm, ZylonLLM) else llm.metadata
        )
        return llm_metadata  # type: ignore[no-any-return]
