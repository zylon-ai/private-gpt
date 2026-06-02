import logging

from injector import inject, singleton
from llama_index.core.embeddings import BaseEmbedding

from private_gpt.components.embedding.discovery import get_embedding_models
from private_gpt.components.embedding.factories.factory import EmbeddingFactoryRegistry
from private_gpt.components.embedding.registry import EmbeddingRegistry
from private_gpt.components.model_discovery.service import are_distinct_api_bases
from private_gpt.settings.settings import EmbeddingModelConfig, Settings

logger = logging.getLogger(__name__)


@singleton
class EmbeddingComponent:
    @inject
    def __init__(self, settings: Settings, embed_registry: EmbeddingRegistry) -> None:
        self.registry = embed_registry
        self.settings = settings

        self.factory_registry = EmbeddingFactoryRegistry(settings)
        self.embed_models: dict[str, EmbeddingModelConfig] = {
            model.name: model
            for model in self.settings.models
            if model.type == "embedding" and model.enabled
        }
        self._default_model_id = settings.embedding.default_model

        self._initialize_models()

    def _discover_models(self) -> list[EmbeddingModelConfig]:
        if not self.settings.embedding.auto_discover_models:
            return []

        api_base = (
            self.settings.openai.embedding_api_base or self.settings.openai.api_base
        )
        api_key = self.settings.openai.embedding_api_key or self.settings.openai.api_key
        return get_embedding_models(
            api_base,
            api_key,
            force_model_kind=are_distinct_api_bases(
                self.settings.openai.api_base,
                self.settings.openai.embedding_api_base,
            ),
        )

    def _initialize_models(self) -> None:
        for model in self._discover_models():
            if model.name not in self.embed_models:
                self.embed_models[model.name] = model

        if not self.embed_models:
            logger.warning(
                "No embedding models are configured, "
                "or it was impossible to discover any models from the API"
            )

        for model_id, model_config in self.embed_models.items():
            try:
                logger.info(
                    "Initializing embedding model '%s' with mode='%s'",
                    model_id,
                    model_config.mode,
                )

                factory = self.factory_registry.get_factory(model_config.mode)
                instance = factory.create_embedding(model_config)

                alias = instance.alias
                aliases = [alias] if alias and alias != model_id else []
                if model_id == self._default_model_id:
                    aliases.append(EmbeddingRegistry.default())

                self.registry.register(model_id, instance.embedding, aliases=aliases)

                logger.info("Successfully registered embedding model '%s'", model_id)

            except Exception as e:
                logger.error("Failed to initialize model '%s': %s", model_id, e)
                raise

        if not self._default_model_id and self.embed_models:
            self._default_model_id = next(iter(self.embed_models))
            logger.warning(
                "No default embedding model configured. Auto-selecting: '%s'",
                self._default_model_id,
            )

        if self._default_model_id:
            if not self.registry.get(self._default_model_id) and not self.registry.get(
                EmbeddingRegistry.default()
            ):
                raise ValueError(
                    f"Default model '{self._default_model_id}' not found in registered models"
                )

            logger.info("Set default model to '%s'", self._default_model_id)

    def get_embed(self, model_id: str | None = None) -> BaseEmbedding:
        target_model = model_id or self._default_model_id

        if not target_model:
            raise ValueError(
                "No embedding model specified and no models are configured"
            )

        embed = self.registry.get(target_model)
        if not embed:
            available = self.registry.get_all_aliases()
            raise ValueError(
                f"Embedding model '{target_model}' not found. Available: {available}"
            )

        return embed

    def get_alias(self, model_id: str | None = None) -> str | None:
        target_model = model_id or self._default_model_id
        if not target_model:
            return None
        aliases = self.registry.get_aliases(target_model)
        return aliases.pop() if aliases else None

    def get_config(self, model_id: str | None = None) -> EmbeddingModelConfig:
        target_model = model_id or self._default_model_id

        if not target_model:
            raise ValueError(
                "No embedding model specified and no models are configured"
            )

        model_config = self.embed_models.get(target_model)
        if not model_config:
            available = list(self.embed_models.keys())
            raise ValueError(
                f"Embedding model config '{target_model}' not found. Available: {available}"
            )

        return model_config

    @property
    def embedding_model(self) -> BaseEmbedding:
        return self.get_embed()

    @property
    def alias(self) -> str | None:
        return self.get_alias()

    @property
    def config(self) -> EmbeddingModelConfig:
        return self.get_config()
