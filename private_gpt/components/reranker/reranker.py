import logging
from typing import (  # noqa: UP035, we need to keep the consistence with llamaindex
    List,
)

from injector import inject, singleton
from llama_index.core.bridge.pydantic import Field
from llama_index.core.indices.postprocessor import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class RerankerComponent(BaseNodePostprocessor):
    """Reranker component.

    - mode: Reranker mode.
    - enabled: Reranker enabled.

    """

    nodePostPorcesser: BaseNodePostprocessor = Field(
        description="BaseNodePostprocessor class."
    )

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
                    from private_gpt.components.reranker.flagembedding_reranker import (
                        FlagEmbeddingRerankerComponent,
                    )
                except ImportError as e:
                    raise ImportError(
                        "Local dependencies not found, install with `poetry install --extras reranker-flagembedding`"
                    ) from e

                nodePostPorcesser = FlagEmbeddingRerankerComponent(settings)

            case _:
                raise ValueError(
                    "Reranker mode not supported, currently only support flagembedding."
                )

        super().__init__(
            nodePostPorcesser=nodePostPorcesser,
        )

    @classmethod
    def class_name(cls) -> str:
        return "Reranker"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],  # noqa: UP006
        query_bundle: QueryBundle | None = None,
    ) -> List[NodeWithScore]:  # noqa: UP006
        return self.nodePostPorcesser._postprocess_nodes(nodes, query_bundle)
