import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.data_structs import IndexDict
from llama_index.core.indices.base import BaseIndex

from private_gpt.artifact_index.base_artifact_index import (
    ArtifactIndexStatus,
    BaseArtifactIndex,
)
from private_gpt.celery.notify import ProgressStatus
from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.ingest.ingest_component import IngestComponent
from private_gpt.components.ingest.parse_component import ParseComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.paths import local_data_path
from private_gpt.server.ingest.model import IngestedDoc

logger = logging.getLogger(__name__)


class VectorArtifactIndex(BaseArtifactIndex):
    _vector_kwargs: dict[str, Any]

    def __init__(
        self,
        collection: str,
        artifact: str,
        node_store_component: NodeStoreComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        ingest_component: IngestComponent,
        parse_component: ParseComponent,
        embed_model_id: str | None = None,
    ) -> None:
        super().__init__(collection, artifact, node_store_component)
        self.vector_store_component = vector_store_component
        self.embedding_component = embedding_component
        self.ingest_component = ingest_component
        self.parse_component = parse_component
        self.embed_model_id = embed_model_id
        self._vector_kwargs = {
            "show_progress": False,
            "use_async": False,
            "insert_batch_size": 512,
        }

    def _load_plain_index(self) -> BaseIndex[IndexDict]:
        """Loads index from storage in the most basic form: with no service context."""
        return load_index_from_storage(
            index_id=self.index_id(),
            storage_context=StorageContext.from_defaults(
                vector_store=self.vector_store_component.vector_store(self.collection),
                index_store=self.node_store_component.index_store(self.collection),
            ),
            **self._vector_kwargs,
        )

    def index_id(self) -> str:
        return self.index_id_from_artifact(self.artifact)

    @staticmethod
    def index_id_from_artifact(artifact: str) -> str:
        return artifact + "-vector"

    @staticmethod
    def artifact_from_index_id(index_id: str) -> str | None:
        if not index_id.endswith("-vector"):
            return None
        return index_id.split("-")[0]

    def initialize(self) -> None:
        """Initialize a vector index for the given artifact.

        If the index already exists, nothing is done.
        If not, a new one will be created empty and persisted to the storage context.
        """
        logger.info("Initializing vector index for artifact: %s", self.artifact)
        try:
            self._load_plain_index()
            logger.info("Vector index was already initialized, nothing to do")
        except ValueError:
            # There is no index in the storage context, creating a new one
            logger.info("Initializing a new empty vector index")

            index = VectorStoreIndex.from_documents(
                documents=[],
                storage_context=StorageContext.from_defaults(
                    vector_store=self.vector_store_component.vector_store(
                        self.collection
                    ),
                    index_store=self.node_store_component.index_store(self.collection),
                ),
                **self._vector_kwargs,
            )

            index.summary = ArtifactIndexStatus.INITIALIZED.value
            index.set_index_id(self.index_id())
            # TODO take this to the node store component
            index.storage_context.persist(persist_dir=local_data_path / self.collection)
            # Set index as ready
            logger.info("Finished empty vector index creation for = %s", self.artifact)

    # TODO pass the SummaryIndex as a parameter to avoid duplicating the dependency
    def populate(
        self,
        file_data: Path,
        file_metadata: dict[str, Any] | None = None,
        notify: Callable[[ProgressStatus], None] = lambda x: None,
        use_async: bool = False,
    ) -> list[IngestedDoc]:
        """Populate the vector index for the given artifact.

        Uses an already existing index to obtain the data.
        """
        logger.info("Populating vector index for artifact = %s", self.artifact)

        # Load vector index
        index = load_index_from_storage(
            index_id=self.index_id(),
            storage_context=StorageContext.from_defaults(
                vector_store=self.vector_store_component.vector_store(self.collection),
                index_store=self.node_store_component.index_store(self.collection),
            ),
            embed_model=self.embedding_component.get_embed(self.embed_model_id),
            transformations=[],
            **self._vector_kwargs,
        )

        logger.info("Populating vector index for artifact = %s", self.artifact)

        # Step 1. Retrieve file information and validate
        file_info, _, warnings = self.parse_component.load_and_validate_file(
            file_data=file_data,
            file_metadata=file_metadata,
            notify=notify,
        )

        # Step 2. Parse file into nodes
        nodes = self.ingest_component.parse_file_into_nodes(
            artifact=self.artifact,
            collection=self.collection,
            file_info=file_info,
            file_metadata=file_metadata,
            notify=notify,
            warnings=warnings,
        )

        # Step 3. Load index
        self.ingest_component.load_index(
            artifact=self.artifact,
            collection=self.collection,
            index=index,
            index_id=self.index_id(),
            nodes=nodes,
            notify=notify,
            warnings=warnings,
            use_async=use_async,
        )

        logger.info("Finished vector index population for = %s", self.artifact)

        return [IngestedDoc.from_document(node) for node in nodes[:1]]
