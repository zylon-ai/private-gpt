import logging
import os
import tempfile
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, AnyStr, BinaryIO

from injector import inject, singleton

from private_gpt.artifact_index.base_artifact_index import (
    ArtifactIndexStatus,
)
from private_gpt.artifact_index.vector_artifact_index import VectorArtifactIndex
from private_gpt.celery.notify import ProgressStatus
from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.ingest.ingest_component import IngestComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.readers.nodes import TreeNode
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.server.ingest.model import IngestedDoc
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class IngestService:
    @inject
    def __init__(
        self,
        settings: Settings,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
        ingest_component: IngestComponent,
    ) -> None:
        self.settings = settings
        self.vector_store_component = vector_store_component
        self.node_store_component = node_store_component
        self.embedding_component = embedding_component
        self.ingest_component = ingest_component

    def initialize_artifact_indices(self, collection: str, artifact: str) -> None:
        # Initialize Vector Artifact index
        vector_artifact_index = VectorArtifactIndex(
            collection=collection,
            artifact=artifact,
            node_store_component=self.node_store_component,
            vector_store_component=self.vector_store_component,
            embedding_component=self.embedding_component,
            ingest_component=self.ingest_component,
        )
        vector_artifact_index.initialize()

    def populate_vector_index(
        self,
        collection: str,
        artifact: str,
        file_data: Path,
        file_metadata: dict[str, Any] | None = None,
        notify: Callable[[ProgressStatus], None] = lambda x: None,
        use_async: bool = False,
    ) -> list[IngestedDoc]:
        """Populate the vector index with the documents in the summary index.

        Throws a ValueError if the index or its dependencies are not initialized.
        Throws a NotReadyException if the summary index is being populated.
        """
        logger.info("Populating vector index for artifact: %s", artifact)

        # Check if the vector index is initialized
        vector_artifact_index = VectorArtifactIndex(
            collection=collection,
            artifact=artifact,
            node_store_component=self.node_store_component,
            vector_store_component=self.vector_store_component,
            embedding_component=self.embedding_component,
            ingest_component=self.ingest_component,
        )
        if vector_artifact_index.status() == ArtifactIndexStatus.NOT_INITIALIZED:
            # Throw an error if the index is not initialized
            raise ValueError(
                f"Vector index for artifact={artifact} is not initialized. "
                f"Please initialize it first."
            )

        # If the summary index is ready, populate the vector index.
        return vector_artifact_index.populate(
            file_data=file_data,
            file_metadata=file_metadata,
            notify=notify,
            use_async=use_async,
        )

    def data_path_from_data(
        self,
        file_data: AnyStr,
        extension: str | None = None,
    ) -> Path:
        logger.debug("Got file data of size=%s to ingest", len(file_data))
        # llama-index mainly supports reading from files, so
        # we have to create a tmp file to read for it to work
        # delete=False to avoid a Windows 11 permission error.
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
            path_to_tmp = Path(tmp.name)
            if isinstance(file_data, bytes):
                path_to_tmp.write_bytes(file_data)
            else:
                path_to_tmp.write_text(str(file_data))

            return path_to_tmp

    def data_path_from_bin_data(
        self,
        raw_file_data: BinaryIO,
        extension: str | None = None,
    ) -> Path:
        logger.debug("Ingesting binary data")
        file_data = raw_file_data.read()
        return self.data_path_from_data(file_data, extension)

    @classmethod
    @contextmanager
    def temporary_file(
        cls, data_path_fn: Callable[[], Path]
    ) -> Generator[Path, None, None]:
        """Create a temporary file and ensure it's deleted after use.

        Args:
            data_path_fn: Function that returns the path to the temporary file

        Yields:
            Path to the temporary file
        """
        tmp_path: Path | None = None
        try:
            tmp_path = data_path_fn()
            yield tmp_path
        finally:
            try:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {tmp_path}: {e}")

    def bulk_ingest(
        self, collection: str, files: list[tuple[Path, str, dict[str, Any]]]
    ) -> list[IngestedDoc]:
        logger.info("Ingesting files =%s", [f[1] for f in files])

        ingested_documents = []
        for file in files:
            data_path, artifact, metadata = file
            # Check if the artifact index is initialized
            self.initialize_artifact_indices(
                collection=collection,
                artifact=artifact,
            )
            # Populate indexes
            ingested_documents.extend(
                self.populate_vector_index(
                    collection=collection,
                    artifact=artifact,
                    file_data=data_path,
                    file_metadata=metadata,
                )
            )
        return ingested_documents

    def get_ingested_files(self, collection: str) -> Generator[IngestedDoc, None, None]:
        # Get the list of artifacts
        artifact_ids = self.node_store_component.get_list_of_artifact_ids(collection)

        # Retrieve root nodes for each artifact
        ingested_docs = 0
        for artifact in artifact_ids:
            nodes = self.node_store_component.filtered_nodes(
                collection,
                artifacts=[artifact],
                limit=1,
            )
            if not nodes:
                return

            sample_node = nodes[0]
            if isinstance(sample_node, TreeNode) and sample_node.root_id:
                # Instead of return that node, retrieve root node
                root_nodes = self.node_store_component.get_nodes(
                    collection,
                    node_ids=[sample_node.root_id],
                    limit=1,
                )
                if root_nodes:
                    nodes = root_nodes

            for n in nodes:
                yield IngestedDoc.from_document(n)
                ingested_docs += 1

        # Log the number of ingested documents
        logger.debug("Found count=%s ingested documents", ingested_docs)

    def delete(self, collection: str, artifact: str, force: bool = False) -> None:
        """Delete the context of the artifact."""
        logger.info(
            "Deleting the ingested artifact: %s",
            artifact,
        )

        # Check if the vector index is initialized
        vector_artifact_index = VectorArtifactIndex(
            collection=collection,
            artifact=artifact,
            node_store_component=self.node_store_component,
            vector_store_component=self.vector_store_component,
            embedding_component=self.embedding_component,
            ingest_component=self.ingest_component,
        )

        try:
            vector_artifact_index.populated_or_error()
        except Exception as e:
            if force:
                logger.warning(
                    "Force delete was requested, deleting the artifact index: %s",
                    artifact,
                )
            else:
                raise e

        vector_artifact_index.delete()
