import abc
import asyncio
import logging
import math
from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.data_structs.data_structs import IndexStruct
from llama_index.core.indices.base import BaseIndex
from llama_index.core.ingestion import arun_transformations, run_transformations
from llama_index.core.schema import (
    BaseNode,
    IndexNode,
    MetadataMode,
    TransformComponent,
)
from llama_index.core.utils import get_tqdm_iterable, iter_batch

from private_gpt.celery.notify import NotifyProtocol
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.paths import local_data_path
from private_gpt.settings.settings import settings
from private_gpt.utils.concurrency import bounded_concurrent_execute

if TYPE_CHECKING:
    from collections.abc import Coroutine

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if settings().server.debug_mode else logging.INFO)


class ArtifactIndexStatus(Enum):
    NOT_INITIALIZED = "not-initialized"
    INITIALIZED = "initialized"
    POPULATED = "populated"


class IndexNotReadyException(Exception):
    """Exception raised when an index is not ready to be used."""

    pass


class BaseArtifactIndex(abc.ABC):
    def __init__(
        self,
        collection: str,
        artifact: str,
        node_store_component: NodeStoreComponent,
    ) -> None:
        self.collection = collection
        self.artifact = artifact
        self.node_store_component = node_store_component

    @abc.abstractmethod
    def initialize(self) -> None:
        pass

    @abc.abstractmethod
    def index_id(self) -> str:
        pass

    @staticmethod
    @abc.abstractmethod
    def index_id_from_artifact(artifact: str) -> str:
        pass

    @abc.abstractmethod
    def _load_plain_index(self) -> BaseIndex[Any]:
        pass

    def populated_or_error(self) -> None:
        """Throws a typed error if the index is not populated to be used.

        Throws a ValueError if the index is not initialized.
        Throws a NotReadyException if the index is being populated.
        """
        if self.status() == ArtifactIndexStatus.NOT_INITIALIZED:
            # Throw an error if the index is not initialized
            raise ValueError(
                f"{type(self).__name__} for artifact={self.artifact} is not initialized."
            )
        elif self.status() is not ArtifactIndexStatus.POPULATED:
            # Throw an error if the index is not ready
            raise IndexNotReadyException(
                f"{type(self).__name__} for artifact={self.artifact} is being populated."
            )

    async def apopulated_or_error(self) -> None:
        """Throws a typed error if the index is not populated to be used.

        Throws a ValueError if the index is not initialized.
        Throws a NotReadyException if the index is being populated.
        """
        current_status = await self.astatus()
        if current_status == ArtifactIndexStatus.NOT_INITIALIZED:
            # Throw an error if the index is not initialized
            raise ValueError(
                f"{type(self).__name__} for artifact={self.artifact} is not initialized."
            )
        elif current_status is not ArtifactIndexStatus.POPULATED:
            # Throw an error if the index is not ready
            raise IndexNotReadyException(
                f"{type(self).__name__} for artifact={self.artifact} is being populated."
            )

    def delete(self) -> None:
        nodes_to_delete = self.node_store_component.filtered_nodes(
            collection=self.collection,
            artifacts=[self.artifact],
        )
        node_ids = [node.node_id for node in nodes_to_delete]

        index = self._load_plain_index()

        ref_doc_ids = set()
        for node_id in node_ids:
            try:
                document = index.docstore.get_document(node_id)
                if document and document.ref_doc_id:
                    ref_doc_ids.add(document.ref_doc_id)
            except ValueError:
                # Node not found, may have been deleted already
                pass

        # Iterate every ref doc id and delete it from index
        for ref_doc_id in ref_doc_ids:
            index.delete_ref_doc(ref_doc_id, delete_from_docstore=False)

        # Iterate every node id and delete it from index
        index.delete_nodes(node_ids)

        # Revert status to initialized
        index.summary = ArtifactIndexStatus.INITIALIZED.value
        # TODO take this to the node store component
        index.storage_context.persist(persist_dir=local_data_path / self.collection)

    def status(self) -> ArtifactIndexStatus:
        """Gets the status of the Index."""
        try:
            index = load_index_from_storage(
                index_id=self.index_id(),
                storage_context=StorageContext.from_defaults(
                    index_store=self.node_store_component.index_store(self.collection),
                ),
            )
            return ArtifactIndexStatus(index.summary)
        except ValueError:
            return ArtifactIndexStatus.NOT_INITIALIZED

    async def astatus(self) -> ArtifactIndexStatus:
        """Gets the status of the Index."""
        try:
            index = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: load_index_from_storage(
                    index_id=self.index_id(),
                    storage_context=StorageContext.from_defaults(
                        index_store=self.node_store_component.index_store(
                            self.collection
                        ),
                    ),
                ),
            )
            return ArtifactIndexStatus(index.summary)
        except ValueError:
            return ArtifactIndexStatus.NOT_INITIALIZED


IS = TypeVar("IS", bound=IndexStruct)


class ExtendIndex(Generic[IS]):
    """Extends a base index to allow to save all documents in one step.

    This is necessary to apply transformations to the documents before saving them
    as an atomic operation. If we don't do this, each document will be saved
    individually and the transformations will be applied to each one separately.
    """

    _embed_size: int
    _insert_batch_size: int

    def __init__(
        self,
        source: BaseIndex[IS],
        embed_size: int,
        transformations: list[TransformComponent] | None = None,
        max_truncate_length: int | None = None,
    ) -> None:
        self._source = source
        self._embed_size = embed_size
        self._source._transformations = transformations or []
        self._insert_batch_size = (
            self._source._insert_batch_size
            if hasattr(self._source, "_insert_batch_size")
            else 512
        )
        self._max_truncate_length = max_truncate_length

    def insert(
        self,
        sources: Sequence[BaseNode],
        notify: NotifyProtocol | None = None,
        **insert_kwargs: Any,
    ) -> Sequence[BaseNode]:
        """Insert a list of documents into the index with transformation.

        Args:
            sources (List[BaseNode]): List of sources to insert.
            notify (NotifyProtocol): Notify protocol to send progress updates.
            insert_kwargs (Any): Additional arguments for node insertion.

        Returns:
            List[BaseNode]: List of nodes created from documents.
        """
        show_progress = self._source._show_progress
        with self._source._callback_manager.as_trace("insert"):
            nodes = run_transformations(
                sources,
                self._source._transformations,
                show_progress=show_progress,
            )

            # Generate batch size. If len(nodes) is less than the batch size, we will
            # split into smaller batches (nearest power of 2) to avoid inserting
            # a single batch with a single document.
            num_nodes = len(nodes)
            exponent = math.floor(math.log(num_nodes, 2))
            potential_batch_size = 2**exponent

            if potential_batch_size > self._insert_batch_size:
                potential_batch_size = self._insert_batch_size

            # Generate batched insertions
            batches = list(iter_batch(nodes, potential_batch_size))
            queue_with_progress = enumerate(
                get_tqdm_iterable(
                    batches,
                    show_progress,
                    f"Inserting nodes into {self._source.index_id}",
                )
            )
            for i, nodes_batch in queue_with_progress:
                if isinstance(self._source, VectorStoreIndex):
                    logger.debug(
                        "Getting embeddings for batch %d/%d",
                        i + 1,
                        len(batches),
                    )
                    nodes_batch = self._get_node_with_embedding(
                        nodes_batch, show_progress
                    )
                    logger.debug(
                        "Got embeddings for batch %d/%d",
                        i + 1,
                        len(batches),
                    )

                logger.debug(
                    "Inserting batch %d/%d into index %s",
                    i + 1,
                    len(batches),
                    self._source.index_id,
                )
                self._source.insert_nodes(
                    nodes_batch, show_progress=show_progress, **insert_kwargs
                )
                logger.debug(
                    "Inserted batch %d/%d into index %s",
                    i + 1,
                    len(batches),
                    self._source.index_id,
                )

                # Notify progress
                if notify:
                    notify(percentage=(i + 1) / len(batches) * 100)

        return nodes

    async def ainsert(
        self,
        sources: Sequence[BaseNode],
        notify: NotifyProtocol | None = None,
        **insert_kwargs: Any,
    ) -> Sequence[BaseNode]:
        """Insert a list of documents into the index with transformation.

        Args:
            sources (List[BaseNode]): List of sources to insert.
            notify (NotifyProtocol): Notify protocol to send progress updates.
            insert_kwargs (Any): Additional arguments for node insertion.

        Returns:
            List[BaseNode]: List of nodes created from documents.
        """
        show_progress = self._source._show_progress
        with self._source._callback_manager.as_trace("insert"):
            nodes = await arun_transformations(
                sources,
                self._source._transformations,
                show_progress=show_progress,
            )

            # Generate batch size. If len(nodes) is less than the batch size, we will
            # split into smaller batches (nearest power of 2) to avoid inserting
            # a single batch with a single document.
            num_nodes = len(nodes)
            exponent = math.floor(math.log(num_nodes, 2))
            potential_batch_size = 2**exponent

            if potential_batch_size > self._insert_batch_size:
                potential_batch_size = self._insert_batch_size

            # Thread-safe counter for progress tracking
            _progress_lock = asyncio.Lock()
            _completed_batches: int = 0

            async def ingest(
                batch_index: int, total_batches: int, nodes_batch: Sequence[BaseNode]
            ) -> None:
                nonlocal _progress_lock, _completed_batches

                if isinstance(self._source, VectorStoreIndex):
                    logger.debug(
                        "Getting embeddings for batch %d/%d",
                        batch_index + 1,
                        total_batches,
                    )
                    nodes_batch = await self._aget_node_with_embedding(
                        nodes_batch, show_progress
                    )
                    logger.debug(
                        "Got embeddings for batch %d/%d",
                        batch_index + 1,
                        total_batches,
                    )

                logger.debug(
                    "Inserting batch %d/%d into index %s",
                    batch_index + 1,
                    total_batches,
                    self._source.index_id,
                )

                async def _ainsert(
                    self: VectorStoreIndex,
                    nodes: Sequence[BaseNode],
                    **insert_kwargs: Any,
                ) -> None:
                    """Insert a document."""
                    await self._async_add_nodes_to_index(
                        self._index_struct, nodes, **insert_kwargs
                    )

                async def ainsert_nodes(
                    self: VectorStoreIndex,
                    nodes_batch: Sequence[BaseNode],
                    **kwargs: Any,
                ) -> None:
                    """Insert nodes into the index."""
                    for (
                        node
                    ) in nodes_batch:  # Fixed: was 'nodes' should be 'nodes_batch'
                        if isinstance(node, IndexNode):
                            try:
                                node.dict()
                            except ValueError:
                                self._object_map[node.index_id] = node.obj
                                node.obj = None

                    logger.info(
                        "Inserting %d nodes into index %s",
                        len(nodes_batch),
                        self.index_id,
                    )
                    with self._callback_manager.as_trace("insert_nodes"):
                        logger.info(
                            "Inserting %d nodes into index %s",
                            len(nodes_batch),
                            self.index_id,
                        )
                        await _ainsert(self, nodes_batch, **kwargs)
                        logger.info(
                            "Inserted %d nodes into index %s",
                            len(nodes_batch),
                            self.index_id,
                        )
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self._storage_context.docstore.add_documents(
                                nodes_batch
                            ),
                        )
                        logger.info(
                            "Added %d nodes to docstore for index %s",
                            len(nodes_batch),
                            self.index_id,
                        )

                if isinstance(self._source, VectorStoreIndex):
                    await ainsert_nodes(
                        self._source,
                        nodes_batch,
                        show_progress=show_progress,
                        **insert_kwargs,
                    )
                else:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._source.insert_nodes(
                            nodes_batch, show_progress=show_progress, **insert_kwargs
                        ),
                    )

                logger.debug(
                    "Inserted batch %d/%d into index %s",
                    batch_index + 1,
                    total_batches,
                    self._source.index_id,
                )

                # Update progress
                logger.info(
                    "Completed batch %d/%d for index %s",
                    batch_index + 1,
                    total_batches,
                    self._source.index_id,
                )
                async with _progress_lock:
                    _completed_batches += 1
                    current_percentage = (_completed_batches / total_batches) * 100

                logger.info(
                    "Progress: %d/%d batches completed (%.2f%%)",
                    _completed_batches,
                    total_batches,
                    current_percentage,
                )

                if notify:
                    if notify:
                        # Run sync function in background thread
                        await asyncio.create_task(
                            asyncio.to_thread(notify, percentage=current_percentage)
                        )
                        logger.info(
                            "Progress notification sent for batch %d/%d",
                            batch_index + 1,
                            total_batches,
                        )

        # Generate batched insertions
        batches = list(iter_batch(nodes, potential_batch_size))
        total_batches = len(batches)

        # Reset global counter
        _completed_batches = 0

        queue_with_progress = enumerate(
            get_tqdm_iterable(
                batches,
                show_progress,
                f"Inserting nodes into {self._source.index_id}",
            )
        )

        tasks: list[Coroutine[Any, Any, None]] = []
        for i, batch in queue_with_progress:
            tasks.append(ingest(i, total_batches, batch))

        # Run all tasks concurrently
        await bounded_concurrent_execute(
            tasks=tasks,
            concurrency_limit=1,
        )

        return nodes

    def _get_node_with_embedding(
        self, nodes: Sequence[BaseNode], show_progress: bool
    ) -> Sequence[BaseNode]:
        """Get the embedding for each node in the list."""
        if not isinstance(self._source, VectorStoreIndex):
            return nodes

        def embed_nodes(
            nodes: Sequence[BaseNode],
            embed_model: BaseEmbedding,
            embed_model_size: int,
            show_progress: bool = False,
        ) -> dict[str, list[float]]:
            """Get embeddings of the given nodes, run embedding model if necessary.

            Args:
                nodes (Sequence[BaseNode]): The nodes to embed.
                embed_model (BaseEmbedding): The embedding model to use.
                embed_model_size (int): The size of the embedding model.
                show_progress (bool): Whether to show progress bar.

            Returns:
                Dict[str, List[float]]: A map from node id to embedding.
            """
            id_to_embed_map: dict[str, list[float]] = {}

            texts_to_embed = []
            ids_to_embed = []
            for node in nodes:
                if node.embedding is None:
                    ids_to_embed.append(node.node_id)
                    content = node.get_content(metadata_mode=MetadataMode.EMBED)
                    if content:
                        # We want to prevent to send too much text to the model
                        # if we know that the model has a limit
                        truncate_content = (
                            content[: self._max_truncate_length]
                            if self._max_truncate_length
                            else content
                        )
                        texts_to_embed.append(truncate_content)
                    else:
                        logger.warning("Node %s has no content, skipping", node.node_id)
                else:
                    id_to_embed_map[node.node_id] = node.embedding

            new_embeddings = embed_model.get_text_embedding_batch(
                texts_to_embed, show_progress=show_progress
            )

            for new_id, text_embedding in zip(
                ids_to_embed, new_embeddings, strict=False
            ):
                id_to_embed_map[new_id] = text_embedding

            # Find nodes that were not embedded
            for node_id in ids_to_embed:
                if node_id not in id_to_embed_map:
                    logger.warning("Node %s was not embedded by the model", node_id)
                    id_to_embed_map[node_id] = [0] * embed_model_size

            return id_to_embed_map

        def _get_node_with_embedding(
            index: VectorStoreIndex,
            nodes: Sequence[BaseNode],
            show_progress: bool = False,
        ) -> list[BaseNode]:
            """Get tuples of id, node, and embedding.

            Allows us to store these nodes in a vector store.
            Embeddings are called in batches.

            """
            id_to_embed_map = embed_nodes(
                nodes, index._embed_model, self._embed_size, show_progress=show_progress
            )

            results = []
            for node in nodes:
                if node.node_id not in id_to_embed_map:
                    raise ValueError(
                        f"Node {node.node_id} was not embedded by the model."
                    )

                embedding = id_to_embed_map[node.node_id]
                if embedding is None or len(embedding) == 0:
                    logger.warning(
                        "Node %s has an empty embedding, skipping", node.node_id
                    )
                    continue
                result = node.model_copy()
                result.embedding = embedding
                results.append(result)
            return results

        return _get_node_with_embedding(
            self._source, nodes, show_progress=show_progress
        )

    async def _aget_node_with_embedding(
        self, nodes: Sequence[BaseNode], show_progress: bool
    ) -> Sequence[BaseNode]:
        """Get the embedding for each node in the list."""
        if not isinstance(self._source, VectorStoreIndex):
            return nodes

        async def aembed_nodes(
            nodes: Sequence[BaseNode],
            embed_model: BaseEmbedding,
            embed_model_size: int,
            show_progress: bool = False,
        ) -> dict[str, list[float]]:
            """Get embeddings of the given nodes, run embedding model if necessary.

            Args:
                nodes (Sequence[BaseNode]): The nodes to embed.
                embed_model (BaseEmbedding): The embedding model to use.
                embed_model_size (int): The size of the embedding model.
                show_progress (bool): Whether to show progress bar.

            Returns:
                Dict[str, List[float]]: A map from node id to embedding.
            """
            id_to_embed_map: dict[str, list[float]] = {}

            texts_to_embed = []
            ids_to_embed = []
            for node in nodes:
                if node.embedding is None:
                    ids_to_embed.append(node.node_id)
                    content = node.get_content(metadata_mode=MetadataMode.EMBED)
                    if content:
                        # We want to prevent to send too much text to the model
                        # if we know that the model has a limit
                        truncate_content = (
                            content[: self._max_truncate_length]
                            if self._max_truncate_length
                            else content
                        )
                        texts_to_embed.append(truncate_content)
                    else:
                        logger.warning("Node %s has no content, skipping", node.node_id)
                else:
                    id_to_embed_map[node.node_id] = node.embedding

            new_embeddings = await embed_model.aget_text_embedding_batch(
                texts_to_embed, show_progress=show_progress
            )

            for new_id, text_embedding in zip(
                ids_to_embed, new_embeddings, strict=False
            ):
                id_to_embed_map[new_id] = text_embedding

            # Find nodes that were not embedded
            for node_id in ids_to_embed:
                if node_id not in id_to_embed_map:
                    logger.warning("Node %s was not embedded by the model", node_id)
                    id_to_embed_map[node_id] = [0] * embed_model_size

            return id_to_embed_map

        async def _aget_node_with_embedding(
            index: VectorStoreIndex,
            nodes: Sequence[BaseNode],
            show_progress: bool = False,
        ) -> list[BaseNode]:
            """Get tuples of id, node, and embedding.

            Allows us to store these nodes in a vector store.
            Embeddings are called in batches.

            """
            id_to_embed_map = await aembed_nodes(
                nodes, index._embed_model, self._embed_size, show_progress=show_progress
            )

            results = []
            for node in nodes:
                if node.node_id not in id_to_embed_map:
                    raise ValueError(
                        f"Node {node.node_id} was not embedded by the model."
                    )

                embedding = id_to_embed_map[node.node_id]
                if embedding is None or len(embedding) == 0:
                    logger.warning(
                        "Node %s has an empty embedding, skipping", node.node_id
                    )
                    continue
                result = node.model_copy()
                result.embedding = embedding
                results.append(result)
            return results

        return await _aget_node_with_embedding(
            self._source, nodes, show_progress=show_progress
        )
