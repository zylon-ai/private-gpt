import asyncio
import contextlib
import logging
import threading
from collections.abc import AsyncGenerator, Generator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from queue import Queue
from typing import Any, ClassVar, Union, cast

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.qdrant import (  # ty:ignore[unresolved-import]
    QdrantVectorStore,
)
from llama_index.vector_stores.qdrant.base import (  # ty:ignore[unresolved-import]
    DEFAULT_DENSE_VECTOR_NAME,
    DEFAULT_SPARSE_VECTOR_NAME_OLD,
    LEGACY_UNNAMED_VECTOR,
)
from llama_index.vector_stores.qdrant.utils import (  # ty:ignore[unresolved-import]
    HybridFusionCallable,
    SparseEncoderCallable,
    relative_score_fusion,
)
from qdrant_client import (  # type: ignore[import-not-found]  # ty:ignore[unresolved-import]
    AsyncQdrantClient,
    QdrantClient,
    models,
)
from qdrant_client.conversions.common_types import (  # type: ignore[import-not-found]  # ty:ignore[unresolved-import]
    OrderBy,
    PayloadSelector,
    QuantizationConfig,
    ReadConsistency,
    Record,
    ShardKeySelector,
)
from qdrant_client.http import (  # ty:ignore[unresolved-import]
    models as rest,  # type: ignore[import-not-found]
)
from qdrant_client.http.models import (  # type: ignore[import-not-found]  # ty:ignore[unresolved-import]
    Filter,
    HasIdCondition,
    Payload,
)
from qdrant_client.local.qdrant_local import (  # type: ignore[import-not-found]  # ty:ignore[unresolved-import]
    QdrantLocal,
)

from private_gpt.components.readers.nodes.utils import metadata_dict_to_tree_node
from private_gpt.utils.retry import retry

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG if settings().server.debug_mode else logging.INFO)

# Multi-tenancy
DEFAULT_GROUP_ID_FIELD = "group_id"

# Retry settings
_MAX_RETRIES = 5
_JITTER = (5, 25)

# Cache for collection initialization
_COLLECTION_INITIALIZED: dict[str, bool] = {}


class PatchedQdrantVectorStore(QdrantVectorStore):
    """QdrantVectorStore impl. that ensures collection creation to prevent crashes.

    This class, PatchedQdrantVectorStore, is a modified version of QdrantVectorStore.
    It includes additional functionality to ensure the creation of a collection
    if it doesn't already exist, preventing crashes when dealing with new collections.

    Args:
        client (QdrantClient)
            The client to interact with Qdrant.
        collection_name (str)
            The name of the collection.
        enable_hybrid_search (bool)
            Flag to enable hybrid search. Default is False.
        embed_dim (int)
            The dimensionality of the embedding vectors. Default is 1024.
        distance (str)
            The distance metric to use. Default is "cosine".
        logical_multitenancy (bool)
            Flag to enable logical multitenancy. Default is False.
        hnsw_m (int)
            The number of neighbors to search. Default is 0.
        hnsw_payload_m (int)
            The number of neighbors to search for payload. Default is 200.
    """

    _logical_multitenancy: bool = PrivateAttr()
    _group_id: str | None = PrivateAttr()
    _group_id_field: str = PrivateAttr(DEFAULT_GROUP_ID_FIELD)

    def __init__(
        self,
        client: QdrantClient,
        aclient: AsyncQdrantClient,
        collection_name: str,
        enable_hybrid: bool = False,
        embed_dim: int = 1024,
        distance: str = "cosine",
        logical_multitenancy: bool = False,
        group_id: str | None = None,
        group_id_field: str | None = None,
        parallel: int = 16,
        batch_size: int = 64,
        hnsw_m: int = 0,
        hnsw_payload_m: int = 16,
        on_disk: bool = True,
        fastembed_sparse_model: str | None = None,
        dense_config: rest.VectorParams | None = None,
        sparse_config: rest.SparseVectorParams | None = None,
        quantization_config: QuantizationConfig | None = None,
        sparse_doc_fn: SparseEncoderCallable | None = None,
        sparse_query_fn: SparseEncoderCallable | None = None,
        hybrid_fusion_fn: HybridFusionCallable | None = None,
        indexes: dict[str, models.KeywordIndexType] | None = None,
    ) -> None:
        # Call parent constructor
        super().__init__(
            # Workaround to avoid calling exist collection
            aclient=object(),
            collection_name=collection_name,
            # Disable hybrid search to avoid client conflicts
            enable_hybrid=False,
            embed_dim=embed_dim,
            distance=models.Distance[distance.upper()],
            bed_sparse_model=None,
            # Disable Qdrant retries to avoid conflicts with the retry decorator
            max_retries=1,
            # Parallel
            parallel=parallel,
            # Batch size
            batch_size=batch_size,
        )
        # Init client after, to avoid to call exist collection
        self._client = client
        self._aclient = aclient

        # Set flag os legacy
        self._legacy_vector_format = True
        if self.enable_hybrid:
            self.dense_vector_name = DEFAULT_DENSE_VECTOR_NAME
            self.sparse_vector_name = DEFAULT_SPARSE_VECTOR_NAME_OLD
        else:
            self.dense_vector_name = LEGACY_UNNAMED_VECTOR

        # Init multi-tenancy
        self._logical_multitenancy = logical_multitenancy
        self._group_id = group_id
        self._group_id_field = group_id_field or DEFAULT_GROUP_ID_FIELD
        if self._logical_multitenancy and not self._group_id:
            raise ValueError("group_id must be provided for logical multitenancy")

        # Init collection
        self._init_collection(
            collection_name,
            enable_hybrid,
            embed_dim,
            models.Distance[distance.upper()],
            logical_multitenancy,
            hnsw_m,
            hnsw_payload_m,
            indexes,
            on_disk,
        )

        # setup hybrid search if enabled
        if enable_hybrid or fastembed_sparse_model is not None:
            self._sparse_doc_fn = sparse_doc_fn or self.get_default_sparse_doc_encoder(
                collection_name, fastembed_sparse_model=fastembed_sparse_model
            )
            self._sparse_query_fn = (
                sparse_query_fn
                or self.get_default_sparse_query_encoder(
                    collection_name, fastembed_sparse_model=fastembed_sparse_model
                )
            )
            self._hybrid_fusion_fn = hybrid_fusion_fn or cast(
                HybridFusionCallable, relative_score_fusion
            )

        self._sparse_config = sparse_config
        self._dense_config = dense_config
        self._quantization_config = quantization_config

    _executor: ClassVar[ThreadPoolExecutor | ProcessPoolExecutor | None] = None

    @classmethod
    def executor(
        cls,
        *args: Any,
        max_workers: int | None = None,
        **kwargs: Any,
    ) -> ThreadPoolExecutor | ProcessPoolExecutor | None:
        """Get executor for parallel processing."""
        if cls._executor is None and max_workers and max_workers > 1:
            cls._executor = ThreadPoolExecutor(*args, **kwargs)
        return cls._executor

    def _init_collection(
        self,
        collection_name: str,
        enable_hybrid_search: bool,
        embed_dim: int,
        distance: models.Distance,
        logical_multitenancy: bool,
        hnsw_m: int,
        hnsw_payload_m: int = 200,
        indexes: dict[str, models.KeywordIndexType] | None = None,
        on_disk: bool = False,
    ) -> None:
        if self._collection_initialized:
            logger.debug("Collection already initialized")
            return

        if collection_name in _COLLECTION_INITIALIZED:
            logger.debug("Collection already initialized")
            self._collection_initialized = True
            return

        if not self._client.collection_exists(collection_name=collection_name):
            from qdrant_client.qdrant_fastembed import (  # ty:ignore[unresolved-import]
                IDF_EMBEDDING_MODELS,
            )

            # Create collection with config
            dense_config = models.VectorParams(
                size=embed_dim, distance=distance, on_disk=on_disk
            )
            sparse_config = self._sparse_config or rest.SparseVectorParams(
                index=rest.SparseIndexParams(on_disk=on_disk),
                modifier=(
                    rest.Modifier.IDF
                    if self.fastembed_sparse_model in IDF_EMBEDDING_MODELS
                    else None
                ),
            )
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    self.dense_vector_name: dense_config,
                }
                if enable_hybrid_search
                else dense_config,
                sparse_vectors_config={self.dense_vector_name: sparse_config}
                if enable_hybrid_search
                else None,
                # Define multi-tenancy configuration if logical_multitenancy is enabled
                hnsw_config=models.HnswConfigDiff(
                    m=hnsw_m, payload_m=hnsw_payload_m, on_disk=on_disk
                ),
                quantization_config=self._quantization_config,
                on_disk_payload=on_disk,
            )
            logger.debug("Collection %s created", collection_name)

            # Create index for group_id
            # Doc: https://arc.net/l/quote/zkmqsfpv
            if logical_multitenancy:
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=self._group_id_field,
                    field_schema=models.KeywordIndexParams(
                        type=models.KeywordIndexType.KEYWORD,
                        is_tenant=True,
                        on_disk=on_disk,
                    ),
                )
                logger.debug(
                    "Index group_id created for collection %s", collection_name
                )

            # Create more indexes to improve performance
            if indexes is not None:
                for index_name, index_type in indexes.items():
                    self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=index_name,
                        field_schema=models.KeywordIndexParams(
                            type=index_type,
                            on_disk=on_disk,
                        ),
                    )
                    logger.debug(
                        "Index %s created for collection %s",
                        index_name,
                        collection_name,
                    )

        self._collection_initialized = True
        if collection_name not in _COLLECTION_INITIALIZED:
            _COLLECTION_INITIALIZED[collection_name] = True

    def _add_missing_logical_multitenancy_filter(
        self, filters: MetadataFilters | None = None
    ) -> MetadataFilters | None:
        """Add the group_id filter to the query filters if it's not already present.

        Since we can use vector store like node store, we need to ensure that
        the group_id filter is present in the filters to ensure
        correct filtering when logical multitenancy is enabled.

        We do:
        1. Check if the group_id filter is already present in the filters.
        2. Add the group_id filter if it's not already present.
        3. Reduce the tree removing MetadataFilters with no filters
           or with only one filter. This is done to ensure that
           the query is not unnecessarily nested since llama-index
           does not handle empty nested filters well.

        Args:
            filters (MetadataFilters | None): The filters to check and update.

        """
        if not self._logical_multitenancy or not self._group_id:
            return filters

        # Ensure that the group_id filter is present in the filters
        filters = filters or MetadataFilters(filters=[], condition=FilterCondition.AND)

        # Recursively check if the group_id filter is already present
        def has_group_id_filter(f: MetadataFilters) -> bool:
            for filt in f.filters:
                if isinstance(filt, MetadataFilter):
                    if (
                        filt.key == self._group_id_field
                        and filt.operator == FilterOperator.EQ
                    ):
                        return True
                elif isinstance(filt, MetadataFilters):
                    return has_group_id_filter(filt)
            return False

        # Add the group_id filter if it's not already present
        if not has_group_id_filter(filters):
            tenancy_filter = MetadataFilter(
                key=self._group_id_field,
                value=self._group_id,
                operator=FilterOperator.EQ,
            )
            if (
                isinstance(filters, MetadataFilters)
                and filters.condition == FilterCondition.AND
            ):
                filters.filters.append(tenancy_filter)
            elif isinstance(filters, MetadataFilters):
                filters = MetadataFilters(
                    filters=[tenancy_filter, filters], condition=FilterCondition.AND
                )
            elif isinstance(filters, MetadataFilter):
                composite_filters: list[MetadataFilter] = (
                    [tenancy_filter, filters] if filters else [tenancy_filter]
                )
                filters = MetadataFilters(
                    filters=composite_filters, condition=FilterCondition.AND
                )

        # Reduce tree removing MetadataFilters with no filters or with only one filter
        def reduce_tree(
            f: Union[MetadataFilter, ExactMatchFilter, "MetadataFilters"]
        ) -> Union[MetadataFilter, ExactMatchFilter, "MetadataFilters"] | None:
            if isinstance(f, MetadataFilter):
                return f
            if len(f.filters) == 0:
                return None
            if len(f.filters) == 1:
                return f.filters[0]

            new_filters = []
            for filter in f.filters:
                new_filter = reduce_tree(filter)
                if new_filter:
                    new_filters.append(new_filter)
            f.filters = new_filters

            return f

        if filters is not None:
            new_filters = []
            for filt in filters.filters:
                if isinstance(filt, MetadataFilters):
                    new_filt = reduce_tree(filt)
                    if new_filt:
                        new_filters.append(new_filt)
                else:
                    new_filters.append(filt)
            filters.filters = new_filters

        return filters

    def _build_query_filter(self, query: VectorStoreQuery) -> Any | None:
        """Override to ensure that the group_id filter is present in the filters."""
        query.filters = self._add_missing_logical_multitenancy_filter(query.filters)
        return super()._build_query_filter(query)

    def _build_points(
        self, nodes: list[BaseNode], sparse_vector_name: str
    ) -> tuple[list[Any], list[str]]:
        """Override to ensure that the group_id is set for logical multitenancy."""
        if self._logical_multitenancy:
            for node in nodes:
                if node.metadata.get(self._group_id_field) is None:
                    node.metadata[self._group_id_field] = self._group_id

        points, ids = super()._build_points(nodes, sparse_vector_name)
        return points, ids

    @staticmethod
    def _process_point(point: Any) -> tuple[BaseNode, str, float]:
        """Process a single point and return the result."""
        payload = cast(Payload, point.payload)
        vector = point.vector
        embedding = None

        if isinstance(vector, dict):
            embedding = vector.get(DEFAULT_DENSE_VECTOR_NAME, vector.get("", None))
        elif isinstance(vector, list):
            embedding = vector

        point_similarity = 1.0
        with contextlib.suppress(AttributeError):
            point_similarity = point.score or 1.0

        node = metadata_dict_to_tree_node(payload)

        if embedding and node.embedding is None:
            node.embedding = embedding

        return node, str(point.id), point_similarity

    def _process_points_in_parallel(
        self, response: list[Any]
    ) -> list[tuple[BaseNode, str, float]]:
        """Process points in parallel using ThreadPoolExecutor."""
        logger.debug(
            "Processing %d points using %d parallel tasks", len(response), self.parallel
        )
        executor = self.executor(max_workers=self.parallel)
        if executor:
            results = list(executor.map(self._process_point, response))
        else:
            results = [self._process_point(point) for point in response]
        logger.debug("Processed %d points", len(results))

        return results

    def parse_to_query_result(self, response: list[Any]) -> VectorStoreQueryResult:
        """Convert vector store response to VectorStoreQueryResult.

        Args:
            response: List[Any]: List of results returned from the vector store.
        """
        results = self._process_points_in_parallel(response)

        nodes = []
        ids = []
        similarities = []

        for node, point_id, similarity in results:
            nodes.append(node)
            ids.append(point_id)
            similarities.append(similarity)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    async def aparse_to_query_result(
        self, response: list[Any]
    ) -> VectorStoreQueryResult:
        """Convert vector store response to VectorStoreQueryResult.

        Args:
            response: List[Any]: List of results returned from the vector store.
        """
        results = await asyncio.to_thread(self._process_points_in_parallel, response)

        nodes = []
        ids = []
        similarities = []

        for node, point_id, similarity in results:
            nodes.append(node)
            ids.append(point_id)
            similarities.append(similarity)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def scroll_all_records(
        self,
        collection_name: str,
        scroll_filter: Filter | None = None,
        limit: int | None = None,
        batch_size: int = 1000,
        order_by: OrderBy | None = None,
        with_payload: bool | Sequence[str] | PayloadSelector = True,
        with_vectors: bool | Sequence[str] = False,
        consistency: ReadConsistency | None = None,
        shard_key_selector: ShardKeySelector | None = None,
        timeout: int | None = None,
    ) -> Generator[list[Record], None, None]:
        counter = 0
        offset = None
        has_more = True

        while has_more:
            current_batch_size = (
                min(batch_size, limit - counter) if limit is not None else batch_size
            )

            if current_batch_size <= 0:
                break

            records, next_offset = self._client.scroll(
                collection_name=collection_name,
                limit=current_batch_size,
                scroll_filter=scroll_filter,
                offset=offset,
                order_by=order_by,
                with_payload=with_payload,
                with_vectors=with_vectors,
                consistency=consistency,
                shard_key_selector=shard_key_selector,
                timeout=timeout,
            )

            if not records:
                break

            yield records
            counter += len(records)

            if next_offset is None or len(records) < current_batch_size:
                has_more = False
            else:
                offset = next_offset

    async def ascroll_all_records(
        self,
        collection_name: str,
        scroll_filter: Filter | None = None,
        limit: int | None = None,
        batch_size: int = 1000,
        order_by: OrderBy | None = None,
        with_payload: bool | Sequence[str] | PayloadSelector = True,
        with_vectors: bool | Sequence[str] = False,
        consistency: ReadConsistency | None = None,
        shard_key_selector: ShardKeySelector | None = None,
        timeout: int | None = None,
    ) -> AsyncGenerator[list[Record]]:
        async def gen() -> AsyncGenerator[list[Record]]:
            counter = 0
            offset = None
            has_more = True

            logger.debug(
                "Starting to scroll all records in collection '%s' with limit %s and batch size %d",
                collection_name,
                limit,
                batch_size,
            )
            while has_more:
                current_batch_size = (
                    min(batch_size, limit - counter)
                    if limit is not None
                    else batch_size
                )
                if current_batch_size == 0:
                    break

                logger.debug(
                    "Preparing to yield records from collection '%s' with offset %s",
                    collection_name,
                    offset,
                )

                records, next_offset = await self._aclient.scroll(
                    collection_name=collection_name,
                    limit=current_batch_size,
                    scroll_filter=scroll_filter,
                    offset=offset,
                    order_by=order_by,
                    with_payload=with_payload,
                    with_vectors=with_vectors,
                    consistency=consistency,
                    shard_key_selector=shard_key_selector,
                    timeout=timeout,
                )

                if not records:
                    break

                logger.debug(
                    "Yielding %d records from collection '%s' with offset %s",
                    len(records),
                    collection_name,
                    offset,
                )

                yield records
                counter += len(records)

                if next_offset is None or len(records) < current_batch_size:
                    has_more = False
                else:
                    offset = next_offset

        return gen()

    @retry(is_async=False, tries=_MAX_RETRIES, jitter=_JITTER, logger=logger)
    def get_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        limit: int | None = None,
    ) -> list[BaseNode]:
        """Override to ensure that the group_id filter is present in the filters."""
        filters = self._add_missing_logical_multitenancy_filter(filters)

        should = []
        if node_ids is not None:
            should = [
                HasIdCondition(
                    has_id=node_ids,
                )
            ]
            # If we pass a node_ids list,
            # we can limit the search to only those nodes
            # or less if limit is provided
            limit = len(node_ids) if limit is None else min(len(node_ids), limit)

        filter: Filter
        if filters is not None:
            filter = self._build_subfilter(filters)
            if filter.should is None:
                filter.should = should
            else:
                filter.should.extend(should)
        else:
            filter = Filter(should=should)

        # If we pass an empty list, Qdrant will not return any results
        filter.must = (
            filter.must
            if filter.must
            and (not isinstance(filter.must, list) or len(filter.must) > 0)
            else None
        )
        filter.should = (
            filter.should
            if filter.should
            and (not isinstance(filter.should, list) or len(filter.should) > 0)
            else None
        )
        filter.must_not = (
            filter.must_not
            if filter.must_not
            and (not isinstance(filter.must_not, list) or len(filter.must_not) > 0)
            else None
        )

        record_queue = Queue[list[Record] | None]()
        node_queue = Queue[Sequence[BaseNode] | None]()
        nodes: list[BaseNode] = []
        num_consumers = self.parallel

        def producer() -> None:
            for records in self.scroll_all_records(
                collection_name=self.collection_name,
                limit=limit,
                scroll_filter=filter,
                with_payload=True,
                with_vectors=True,
            ):
                record_queue.put(records)

            # Signal consumers to stop
            for _ in range(num_consumers):
                record_queue.put(None)

        def consumer() -> None:
            while True:
                records = record_queue.get()
                if records is None:
                    break
                result = self.parse_to_query_result(records)
                if result.nodes:
                    node_queue.put(result.nodes)

            # Signal collector
            node_queue.put(None)

        def collector() -> None:
            consumers_done = 0
            while consumers_done < num_consumers:
                parsed_nodes = node_queue.get()
                if parsed_nodes is None:
                    consumers_done += 1
                else:
                    nodes.extend(parsed_nodes)

        threads: list[threading.Thread] = [threading.Thread(target=producer)]
        threads.extend(
            [threading.Thread(target=consumer) for _ in range(num_consumers)]
        )
        threads.append(threading.Thread(target=collector))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        return nodes

    @retry(is_async=True, tries=_MAX_RETRIES, jitter=_JITTER, logger=logger)
    async def aget_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        limit: int | None = None,
    ) -> list[BaseNode]:
        """Override to ensure that the group_id filter is present in the filters."""
        # TODO: To pass tests: As we don't migrate to async yet, we need to
        # to use the sync method until we migrate to async
        if isinstance(self._client._client, QdrantLocal):
            # If we are using QdrantLocal, we need to use the sync method
            return cast(  # type: ignore[redundant-cast]
                list[BaseNode],
                super().get_nodes(node_ids=node_ids, filters=filters, limit=limit),
            )

        filters = await asyncio.to_thread(
            self._add_missing_logical_multitenancy_filter, filters
        )

        should = []
        if node_ids is not None:
            should = [
                HasIdCondition(
                    has_id=node_ids,
                )
            ]
            # If we pass a node_ids list,
            # we can limit the search to only those nodes
            # or less if limit is provided
            limit = len(node_ids) if limit is None else min(len(node_ids), limit)

        filter: Filter
        if filters is not None:
            filter = await asyncio.to_thread(self._build_subfilter, filters)
            if filter.should is None:
                filter.should = should
            else:
                filter.should.extend(should)
        else:
            filter = Filter(should=should)

        # If we pass an empty list, Qdrant will not return any results
        filter.must = (
            filter.must
            if filter.must
            and (not isinstance(filter.must, list) or len(filter.must) > 0)
            else None
        )
        filter.should = (
            filter.should
            if filter.should
            and (not isinstance(filter.should, list) or len(filter.should) > 0)
            else None
        )
        filter.must_not = (
            filter.must_not
            if filter.must_not
            and (not isinstance(filter.must_not, list) or len(filter.must_not) > 0)
            else None
        )

        queue = asyncio.Queue[list[Record] | None]()
        node_queue = asyncio.Queue[Sequence[BaseNode] | None]()
        nodes: list[BaseNode] = []
        num_consumers = self.parallel

        async def producer() -> None:
            async for records in await self.ascroll_all_records(
                collection_name=self.collection_name,
                limit=limit,
                scroll_filter=filter,
                with_payload=True,
                with_vectors=True,
            ):
                await queue.put(records)

            # Signal the consumer that
            # there are no more records
            for _ in range(num_consumers):
                await queue.put(None)

        async def consumer() -> None:
            while True:
                records = await queue.get()
                if records is None:
                    # No more records to process
                    break
                result = await self.aparse_to_query_result(records)
                if result.nodes:
                    await node_queue.put(result.nodes)

            # Signal the consumer that
            # there are no more records
            await node_queue.put(None)

        async def collector() -> None:
            consumers_done = 0
            while consumers_done < num_consumers:
                parsed_nodes = await node_queue.get()
                if parsed_nodes is None:
                    consumers_done += 1
                else:
                    nodes.extend(parsed_nodes)

        await asyncio.gather(
            producer(), *[consumer() for _ in range(num_consumers)], collector()
        )
        return nodes

    @retry(is_async=False, tries=_MAX_RETRIES, jitter=_JITTER, logger=logger)
    def add(self, nodes: list[BaseNode], **add_kwargs: Any) -> list[str]:
        """Override to add retry logic to the add method."""
        parent: list[str] = super().add(nodes, **add_kwargs)
        return parent

    @retry(is_async=True, tries=_MAX_RETRIES, jitter=_JITTER, logger=logger)
    async def async_add(self, nodes: list[BaseNode], **kwargs: Any) -> list[str]:
        """Override to add retry logic to the async_add method."""
        # TODO: To pass tests: As we don't migrate to async yet, we need to
        # to use the sync method until we migrate to async
        if isinstance(self._client._client, QdrantLocal):
            # If we are using QdrantLocal, we need to use the sync method
            return cast(list[str], super().add(nodes, **kwargs))  # type: ignore[redundant-cast]

        parent: list[str] = await super().async_add(nodes, **kwargs)
        return parent

    @retry(is_async=False, tries=_MAX_RETRIES, jitter=_JITTER, logger=logger)
    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Override to add retry logic to the delete method."""
        super().delete(ref_doc_id, **delete_kwargs)

    @retry(is_async=True, tries=_MAX_RETRIES, jitter=_JITTER, logger=logger)
    async def adelete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Override to add retry logic to the async_delete method."""
        # TODO: To pass tests: As we don't migrate to async yet, we need to
        # to use the sync method until we migrate to async
        if isinstance(self._client._client, QdrantLocal):
            # If we are using QdrantLocal, we need to use the sync method
            super().delete(ref_doc_id, **kwargs)
            return

        await super().adelete(ref_doc_id, **kwargs)

    @retry(is_async=False, tries=_MAX_RETRIES, jitter=_JITTER, logger=logger)
    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:
        """Override to add retry logic to the delete_nodes method."""
        super().delete_nodes(node_ids=node_ids, filters=filters, **delete_kwargs)

    @retry(is_async=True, tries=_MAX_RETRIES, jitter=_JITTER, logger=logger)
    async def adelete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **kwargs: Any,
    ) -> None:
        """Override to add retry logic to the async_delete_nodes method."""
        # TODO: To pass tests: As we don't migrate to async yet, we need to
        # to use the sync method until we migrate to async
        if isinstance(self._client._client, QdrantLocal):
            # If we are using QdrantLocal, we need to use the sync method
            super().delete_nodes(node_ids=node_ids, filters=filters, **kwargs)
            return

        await super().adelete_nodes(node_ids=node_ids, filters=filters, **kwargs)

    @retry(is_async=False, tries=_MAX_RETRIES, jitter=_JITTER, logger=logger)
    def clear(self) -> None:
        """Override to add retry logic to the clear method."""
        super().clear()

    @retry(is_async=True, tries=_MAX_RETRIES, jitter=_JITTER, logger=logger)
    async def aclear(self) -> None:
        """Override to add retry logic to the async_clear method."""
        # TODO: To pass tests: As we don't migrate to async yet, we need to
        # to use the sync method until we migrate to async
        if isinstance(self._client._client, QdrantLocal):
            # If we are using QdrantLocal, we need to use the sync method
            super().clear()
            return

        await super().aclear()

    def _query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): query
            **kwargs: additional keyword arguments to pass to the query

        """
        query_embedding = cast(list[float], query.query_embedding)

        with_payload = kwargs.pop("with_payload", True)
        with_vector = kwargs.pop("with_vectors", False)
        score_threshold = kwargs.pop("score_threshold", None)

        qdrant_filters = kwargs.get("qdrant_filters")
        if qdrant_filters is not None:
            query_filter = qdrant_filters
        else:
            query_filter = cast(Filter, self._build_query_filter(query))

        if query.mode == VectorStoreQueryMode.HYBRID and not self.enable_hybrid:
            raise ValueError(
                "Hybrid search is not enabled. Please build the query with "
                "`enable_hybrid=True` in the constructor."
            )
        elif (
            query.mode == VectorStoreQueryMode.HYBRID
            and self.enable_hybrid
            and self._sparse_query_fn is not None
            and query.query_str is not None
        ):
            sparse_indices, sparse_embedding = self._sparse_query_fn(
                [query.query_str],
            )
            sparse_top_k = query.sparse_top_k or query.similarity_top_k

            sparse_response = self._client.query_batch_points(
                collection_name=self.collection_name,
                requests=[
                    rest.QueryRequest(
                        query=query_embedding,
                        using=self.dense_vector_name,
                        limit=query.similarity_top_k,
                        filter=query_filter,
                        with_payload=with_payload,
                        with_vector=with_vector,
                        score_threshold=score_threshold,
                    ),
                    rest.QueryRequest(
                        query=rest.SparseVector(
                            indices=sparse_indices[0],
                            values=sparse_embedding[0],
                        ),
                        using=self.sparse_vector_name,
                        limit=sparse_top_k,
                        filter=query_filter,
                        with_payload=with_payload,
                        with_vector=with_vector,
                        score_threshold=score_threshold,
                    ),
                ],
            )

            # sanity check
            assert len(sparse_response) == 2
            assert self._hybrid_fusion_fn is not None

            # flatten the response
            return cast(  # type: ignore[redundant-cast]
                VectorStoreQueryResult,
                self._hybrid_fusion_fn(
                    self.parse_to_query_result(sparse_response[0].points),
                    self.parse_to_query_result(sparse_response[1].points),
                    # NOTE: only for hybrid search
                    # (0 for sparse search, 1 for dense search)
                    alpha=query.alpha or 0.5,
                    # NOTE: use hybrid_top_k if provided,
                    # otherwise use similarity_top_k
                    top_k=query.hybrid_top_k or query.similarity_top_k,
                ),
            )
        elif (
            query.mode == VectorStoreQueryMode.SPARSE
            and self.enable_hybrid
            and self._sparse_query_fn is not None
            and query.query_str is not None
        ):
            sparse_indices, sparse_embedding = self._sparse_query_fn(
                [query.query_str],
            )
            sparse_top_k = query.sparse_top_k or query.similarity_top_k

            sparse_response = self._client.query_batch_points(
                collection_name=self.collection_name,
                requests=[
                    rest.QueryRequest(
                        query=rest.SparseVector(
                            indices=sparse_indices[0],
                            values=sparse_embedding[0],
                        ),
                        using=self.sparse_vector_name,
                        limit=sparse_top_k,
                        filter=query_filter,
                        with_payload=with_payload,
                        with_vector=with_vector,
                        score_threshold=score_threshold,
                    ),
                ],
            )

            return self.parse_to_query_result(sparse_response[0].points)
        elif self.enable_hybrid:
            # search for dense vectors only
            hybrid_response = self._client.query_batch_points(
                collection_name=self.collection_name,
                requests=[
                    rest.QueryRequest(
                        query=query_embedding,
                        using=self.dense_vector_name,
                        limit=query.similarity_top_k,
                        filter=query_filter,
                        with_payload=with_payload,
                        with_vector=with_vector,
                        score_threshold=score_threshold,
                    ),
                ],
            )

            return self.parse_to_query_result(hybrid_response[0].points)
        else:
            # Regular non-hybrid search
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                using=self.dense_vector_name,
                limit=query.similarity_top_k,
                query_filter=query_filter,
                with_payload=with_payload,
                with_vectors=with_vector,
                score_threshold=score_threshold,
            )
            return self.parse_to_query_result(response.points)

    @retry(is_async=False, tries=_MAX_RETRIES, jitter=_JITTER, logger=logger)
    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Override to add retry logic to the query method."""
        if query.query_str == "":
            # If we pass a empty parameter, it returns an error
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        parent: VectorStoreQueryResult = self._query(query, **kwargs)
        return parent

    async def _aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Asynchronous method to query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): query
            **kwargs: additional keyword arguments to pass to the query
        """
        query_embedding = cast(list[float], query.query_embedding)

        with_payload = kwargs.pop("with_payload", True)
        with_vector = kwargs.pop("with_vectors", False)
        score_threshold = kwargs.pop("score_threshold", None)

        #  NOTE: users can pass in qdrant_filters (nested/complicated filters)
        #  to override the default MetadataFilters
        qdrant_filters = kwargs.get("qdrant_filters")
        if qdrant_filters is not None:
            query_filter = qdrant_filters
        else:
            # build metadata filters
            query_filter = cast(
                Filter, await asyncio.to_thread(self._build_query_filter, query)
            )

        if query.mode == VectorStoreQueryMode.HYBRID and not self.enable_hybrid:
            raise ValueError(
                "Hybrid search is not enabled. Please build the query with "
                "`enable_hybrid=True` in the constructor."
            )
        elif (
            query.mode == VectorStoreQueryMode.HYBRID
            and self.enable_hybrid
            and self._sparse_query_fn is not None
            and query.query_str is not None
        ):
            sparse_indices, sparse_embedding = await asyncio.to_thread(
                self._sparse_query_fn,
                [query.query_str],
            )
            sparse_top_k = query.sparse_top_k or query.similarity_top_k

            sparse_response = await self._aclient.query_batch_points(
                collection_name=self.collection_name,
                requests=[
                    rest.QueryRequest(
                        query=query_embedding,
                        using=self.dense_vector_name,
                        limit=query.similarity_top_k,
                        filter=query_filter,
                        with_payload=with_payload,
                        with_vector=with_vector,
                        score_threshold=score_threshold,
                    ),
                    rest.QueryRequest(
                        query=rest.SparseVector(
                            indices=sparse_indices[0],
                            values=sparse_embedding[0],
                        ),
                        using=self.sparse_vector_name,
                        limit=sparse_top_k,
                        filter=query_filter,
                        with_payload=with_payload,
                        with_vector=with_vector,
                        score_threshold=score_threshold,
                    ),
                ],
            )

            # sanity check
            assert len(sparse_response) == 2
            assert self._hybrid_fusion_fn is not None

            # flatten the response
            return cast(  # type: ignore[redundant-cast]
                VectorStoreQueryResult,
                self._hybrid_fusion_fn(
                    await self.aparse_to_query_result(sparse_response[0].points),
                    await self.aparse_to_query_result(sparse_response[1].points),
                    alpha=query.alpha or 0.5,
                    # NOTE: use hybrid_top_k if provided, otherwise use similarity_top_k
                    top_k=query.hybrid_top_k or query.similarity_top_k,
                ),
            )
        elif (
            query.mode == VectorStoreQueryMode.SPARSE
            and self.enable_hybrid
            and self._sparse_query_fn is not None
            and query.query_str is not None
        ):
            sparse_indices, sparse_embedding = await asyncio.to_thread(
                self._sparse_query_fn,
                [query.query_str],
            )
            sparse_top_k = query.sparse_top_k or query.similarity_top_k

            sparse_response = await self._aclient.query_batch_points(
                collection_name=self.collection_name,
                requests=[
                    rest.QueryRequest(
                        query=rest.SparseVector(
                            indices=sparse_indices[0],
                            values=sparse_embedding[0],
                        ),
                        using=self.sparse_vector_name,
                        limit=sparse_top_k,
                        filter=query_filter,
                        with_payload=with_payload,
                        with_vector=with_vector,
                    ),
                ],
            )
            return await self.aparse_to_query_result(sparse_response[0].points)
        elif self.enable_hybrid:
            # search for dense vectors only
            hybrid_response = await self._aclient.query_batch_points(
                collection_name=self.collection_name,
                requests=[
                    rest.QueryRequest(
                        query=query_embedding,
                        using=self.dense_vector_name,
                        limit=query.similarity_top_k,
                        filter=query_filter,
                        with_payload=with_payload,
                        with_vector=with_vector,
                        score_threshold=score_threshold,
                    ),
                ],
            )

            return await self.aparse_to_query_result(hybrid_response[0].points)
        else:
            response = await self._aclient.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                using=self.dense_vector_name,
                limit=query.similarity_top_k,
                query_filter=query_filter,
                with_payload=with_payload,
                with_vectors=with_vector,
                score_threshold=score_threshold,
            )

            return await self.aparse_to_query_result(response.points)

    @retry(is_async=True, tries=_MAX_RETRIES, jitter=_JITTER, logger=logger)
    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Override to add retry logic to the async_query method."""
        if query.query_str == "":
            # If we pass a empty parameter, it returns an error
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        # TODO: To pass tests: As we don't migrate to async yet, we need to
        # to use the sync method until we migrate to async
        if isinstance(self._client._client, QdrantLocal):
            # If we are using QdrantLocal, we need to use the sync method
            return cast(VectorStoreQueryResult, super().query(query, **kwargs))  # type: ignore[redundant-cast]

        # Patched parent method to use async methods
        parent: VectorStoreQueryResult = await self._aquery(query, **kwargs)
        return parent
