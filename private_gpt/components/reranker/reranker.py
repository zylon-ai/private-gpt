import logging

from injector import inject, singleton

from private_gpt.paths import models_path
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class RerankerComponent:
    """Reranker component.

    - mode: Reranker mode.
    - enabled: Reranker enabled.

    """

    @inject
    def __init__(self, settings: Settings) -> None:
        if settings.reranker.enabled is False:
            raise ValueError("Reranker component is not enabled.")

        match settings.reranker.mode:
            case "flagembedding":
                logger.info(
                    "Initializing the reranker model in mode=%s", settings.reranker.mode
                )

                try:
                    from FlagEmbedding import FlagReranker  # type: ignore

                    from private_gpt.components.reranker.flagembedding_reranker import (
                        FlagEmbeddingRerankerComponent,
                    )
                except ImportError as e:
                    raise ImportError(
                        "Local dependencies not found, install with `poetry install --extras reranker-flagembedding`"
                    ) from e

                path = models_path / "flagembedding_reranker"

                if settings.flagembedding_reranker is None:
                    raise ValueError("FlagEmbeddingReranker settings is not provided.")

                top_n = settings.flagembedding_reranker.top_n
                cut_off = settings.flagembedding_reranker.cut_off
                flagReranker = FlagReranker(
                    model_name_or_path=path,
                )
                self.nodePostPorcesser = FlagEmbeddingRerankerComponent(
                    top_n=top_n, cut_off=cut_off, reranker=flagReranker
                )

            case _:
                raise ValueError(
                    "Reranker mode not supported, currently only support flagembedding."
                )
