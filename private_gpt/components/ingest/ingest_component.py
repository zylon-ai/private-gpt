import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from injector import inject, singleton
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores import FilterCondition, FilterOperator

from private_gpt.artifact_index.artifact_exception import InvalidFileError
from private_gpt.artifact_index.base_artifact_index import (
    ArtifactIndexStatus,
    ExtendIndex,
)
from private_gpt.celery.notify import ProgressStatus, notify_progress
from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.ingest.fake_progress import calculate_parsing_timing
from private_gpt.components.ingest.ingest_helper import IngestionHelper
from private_gpt.components.ingest.metadata_helper import MetadataChunk, MetadataKeys
from private_gpt.components.ingest.parse_component import ParseComponent
from private_gpt.components.ingest.progress.errors import (
    IngestionLoadErrors,
    IngestionParseErrors,
)
from private_gpt.components.ingest.progress.models import (
    ParseProgressStatus,
    StorageProgressStatus,
)
from private_gpt.components.ingest.utils import FileInfo
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.paths import local_data_path
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from private_gpt.components.ingest.parse_component import FileParseResult

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@singleton
class IngestComponent:
    @inject
    def __init__(
        self,
        settings: Settings,
        node_store_component: NodeStoreComponent,
        llm_component: LLMComponent,
        embedding_component: EmbeddingComponent,
        parse_component: ParseComponent,
    ) -> None:
        self.settings = settings
        self.node_store_component = node_store_component
        self.llm_component = llm_component
        self.embedding_component = embedding_component
        self.parse_component = parse_component

        self._generate_fake_percentage = settings.data.enable_fake_progress
        self._enable_reuse_generated_nodes_before = (
            settings.data.enable_reuse_generated_nodes_before
        )

    def parse_file_into_nodes(
        self,
        artifact: str,
        collection: str,
        file_info: FileInfo,
        file_metadata: dict[str, Any] | None,
        notify: Callable[[ProgressStatus], None] = lambda x: None,
        warnings: list[str] | None = None,
    ) -> list[BaseNode]:
        # Calculate if another file was ingested before
        exists, nodes = self.retrieve_ingested_nodes(artifact, collection, file_info)
        if exists:
            return nodes or []

        # Transform file into nodes
        return self.transform_file_into_nodes(
            artifact=artifact,
            collection=collection,
            file_info=file_info,
            file_metadata=file_metadata,
            notify=notify,
            warnings=warnings,
        )

    def retrieve_ingested_nodes(
        self,
        artifact: str,
        collection: str,
        file_info: FileInfo,
    ) -> tuple[bool, list[BaseNode] | None]:
        """Try to find if exists a node with the same hash in the node store."""
        if not file_info.hash:
            return False, None

        filter_dicts = [
            {
                "key": MetadataKeys.FILE_HASH.value,
                "value": file_info.hash,
                "operator": FilterOperator.EQ,
            }
        ]

        # 1. Try to retrieve nodes from same artifact/collection with the same hash
        nodes = self.node_store_component.filtered_nodes(
            collection=collection,
            artifacts=[artifact],
            filter_dicts=filter_dicts,
            limit=1,
            filter_condition=FilterCondition.AND,
        )
        if nodes:
            logger.info("Artifact is already ingested in the node store. Skipping.")
            return True, []

        # 2. Try to retrieve nodes with same hash but different artifact
        if self._enable_reuse_generated_nodes_before:
            nodes = self.node_store_component.filtered_nodes(
                collection=collection,
                artifacts=None,
                filter_dicts=filter_dicts,
            )
            if nodes:
                logger.info(
                    "Found existing nodes with the same hash and "
                    "different artifact in the node store."
                )

                # Group by artifact and collection
                grouped_nodes: dict[str, list[BaseNode]] = {}
                for node in nodes:
                    n_artifact: str = str(
                        node.metadata.get(MetadataKeys.ARTIFACT_ID.value, None)
                    )
                    n_collection: str = str(
                        node.metadata.get(MetadataKeys.COLLECTION.value, None)
                    )
                    key = f"{n_artifact}_{n_collection}"
                    if key not in grouped_nodes:
                        grouped_nodes[key] = []
                    grouped_nodes[key].append(node)

                # Take only one since all artifacts are the same
                chosen_nodes = grouped_nodes.popitem()[1]

                # Update artifact and collection metadata with the new values
                for node in chosen_nodes:
                    node.metadata[MetadataKeys.ARTIFACT_ID.value] = artifact
                    node.metadata[MetadataKeys.COLLECTION.value] = collection
                return True, nodes

        # 3. No nodes found
        return False, None

    def transform_file_into_nodes(
        self,
        artifact: str,
        collection: str,
        file_info: FileInfo,
        file_metadata: dict[str, Any] | None,
        notify: Callable[[ProgressStatus], None] = lambda x: None,
        warnings: list[str] | None = None,
    ) -> list[BaseNode]:
        """Transform a file into a list of documents.

        This class should be used to transform a file into a list of documents.
        These methods are thread-safe (and multiprocessing-safe).
        """
        interval, jitter = calculate_parsing_timing(
            file_size=file_info.file_size,
            pages=file_info.config.get(MetadataChunk.PAGE.value, 1),
        )
        with notify_progress(
            notify=notify,
            status_class=ParseProgressStatus,
            warnings=warnings,
            generate_fake_percentage=self._generate_fake_percentage,
            generate_fake_percentage_interval_ms=int(interval * 1000)
            if interval
            else None,
            generate_fake_percentage_jitter=jitter,
        ) as notification:
            logger.info("Transforming file into documents: %s", file_info.file_name)

            result: FileParseResult = self.parse_component.file_to_nodes(
                file_info=file_info,
                file_metadata=file_metadata,
                notification=notification,
                warnings=warnings,
            )
            nodes = result.nodes

            max_nodes = self.node_store_component.max_nodes
            if max_nodes and len(nodes) > max_nodes:
                logger.info(
                    "Number of nodes (%d) exceeds the maximum number of nodes (%d)",
                    len(nodes),
                    max_nodes,
                )
                raise InvalidFileError(
                    errors=[IngestionParseErrors.PARSING_FAILURE], warnings=warnings
                )

            for document in nodes:
                # Store artifact and collection metadata
                document.metadata[MetadataKeys.ARTIFACT_ID.value] = artifact
                document.metadata[MetadataKeys.COLLECTION.value] = collection

                # Store LLM and Embedding model metadata
                # to know which models were used to ingest the document
                llm_model = self.llm_component.alias
                if llm_model:
                    document.metadata[MetadataKeys.LLM_MODEL.value] = llm_model
                embed_model = self.embedding_component.get_alias()
                if embed_model:
                    document.metadata[MetadataKeys.EMBED_MODEL.value] = embed_model
                document.metadata.update(file_metadata or {})

                # Store current file hash
                document.metadata[MetadataKeys.FILE_HASH.value] = file_info.hash

            IngestionHelper.exclude_metadata(nodes=nodes, file_metadata=file_metadata)
            logger.info(
                "Finished transforming file into documents: %s", file_info.file_name
            )

        return nodes

    def load_index(
        self,
        artifact: str,
        collection: str,
        index: BaseIndex[Any],
        index_id: str,
        nodes: list[BaseNode],
        notify: Callable[[ProgressStatus], None] = lambda x: None,
        use_async: bool = True,
        warnings: list[str] | None = None,
    ) -> None:
        """Load the index with the given documents."""
        if not nodes:
            # No nodes to insert
            return

        with notify_progress(
            notify=notify,
            status_class=StorageProgressStatus,
            warnings=warnings,
        ) as notify_publisher:
            logger.info("Loading index %s with %d nodes", index_id, len(nodes))
            max_context_window = self.embedding_component.get_config().context_window
            extended_index = ExtendIndex(
                source=index,
                embed_size=self.settings.vectorstore.embed_dim,
                max_truncate_length=max_context_window * 10,  # 10x context window
            )

            # 1. Delete previous nodes, to avoid duplicates
            self.node_store_component.delete_filtered_nodes(
                collection=collection,
                artifacts=[artifact],
            )

            # 2. Insert nodes
            inserted_nodes: Sequence[BaseNode] = []
            if use_async:
                inserted_nodes = asyncio.run(
                    extended_index.ainsert(nodes, notify=notify_publisher)
                )
            else:
                inserted_nodes = extended_index.insert(nodes, notify=notify_publisher)

            if not inserted_nodes:
                raise InvalidFileError(
                    errors=[IngestionLoadErrors.NO_VALID_NODES], warnings=warnings
                )

            index.summary = ArtifactIndexStatus.POPULATED.value
            index.set_index_id(index_id)
            index.storage_context.persist(persist_dir=local_data_path / collection)
            logger.info("Finished loading index %s with %d nodes", index_id, len(nodes))
