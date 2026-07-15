import logging
import re
import typing
from collections.abc import Callable
from typing import Any

from injector import inject, singleton
from llama_index.core.schema import BaseNode
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.index_store.types import BaseIndexStore
from llama_index.core.vector_stores import (
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
)

from private_gpt.components.ingest.metadata_helper import MetadataKeys
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.paths import local_data_path
from private_gpt.settings.settings import Settings
from private_gpt.utils.dependencies import format_missing_dependency_message

logger = logging.getLogger(__name__)

IndexStoreProvider = Callable[[Settings, str], BaseIndexStore]


def _simple_index_store(settings: Settings, collection: str) -> BaseIndexStore:
    del settings
    try:
        return SimpleIndexStore.from_persist_dir(
            persist_dir=str(local_data_path / collection)
        )
    except FileNotFoundError:
        logger.debug("Local index store not found, creating a new one")
        return SimpleIndexStore()


def _postgres_index_store(settings: Settings, collection: str) -> BaseIndexStore:
    try:
        from llama_index.storage.index_store.postgres import (  # ty:ignore[unresolved-import]
            PostgresIndexStore,
        )

        from private_gpt.components.node_store.patched_postgres_kv_store import (
            PatchedPostgresKVStore,
        )
    except ImportError as e:
        raise ImportError(
            format_missing_dependency_message(
                "Postgres node store",
                extras="nodestore-postgres",
            )
        ) from e

    from private_gpt.components.postgres.postgres_client import LazyPostgresFactory

    client = LazyPostgresFactory.get_instance(settings)
    kv_store = PatchedPostgresKVStore(
        client.sync_session,
        client.async_session,
        table_name=collection,
        schema_name="zgptnode",
    )
    return typing.cast(BaseIndexStore, PostgresIndexStore(kv_store))


_PROVIDERS: dict[str, IndexStoreProvider] = {
    "simple": _simple_index_store,
    "postgres": _postgres_index_store,
}


def register_index_store(name: str, provider: IndexStoreProvider) -> None:
    _PROVIDERS[name] = provider


@singleton
class NodeStoreComponent:
    @inject
    def __init__(
        self, settings: Settings, vector_store_component: VectorStoreComponent
    ) -> None:
        self._settings = settings
        self._vector_store_component = vector_store_component

    def index_store(self, collection: str) -> BaseIndexStore:
        provider = _PROVIDERS.get(self._settings.node_store.index_store)
        if provider is None:
            available = ", ".join(sorted(_PROVIDERS)) or "none"
            raise ValueError(
                f"Node store '{self._settings.node_store.index_store}' is not supported. "
                f"Available: {available}"
            )
        return provider(self._settings, collection)

    @property
    def max_nodes(self) -> int | None:
        return self._settings.data.max_num_nodes or None

    def get_nodes(
        self,
        collection: str,
        artifacts: list[str] | None = None,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        limit: int | None = None,
    ) -> list[BaseNode]:
        vector_store = self._vector_store_component.vector_store(collection)
        if vector_store is None:
            raise ValueError(f"Vector store for collection {collection} not found")
        if not hasattr(vector_store, "get_nodes"):
            raise ValueError(
                f"Vector store for collection {collection} does not support get_nodes"
            )

        if artifacts:
            artifact_filters = MetadataFilters(
                filters=[
                    MetadataFilter(key=MetadataKeys.ARTIFACT_ID.value, value=artifact)
                    for artifact in artifacts
                ],
                condition=FilterCondition.OR,
            )
            filters = (
                MetadataFilters(
                    filters=[filters, artifact_filters],
                    condition=FilterCondition.AND,
                )
                if filters
                else artifact_filters
            )

        # Define limit based on node_ids
        if node_ids:
            limit = min(limit, len(node_ids)) if limit is not None else len(node_ids)

        # Set default limit
        # If we don't set it, it will recover only 9999 nodes
        if limit is None:
            limit = self.max_nodes

        return vector_store.get_nodes(node_ids=node_ids, filters=filters, limit=limit)  # type: ignore

    def get_sorted_nodes(
        self,
        collection: str,
        artifacts: list[str] | None = None,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        limit: int | None = None,
    ) -> list[BaseNode]:
        """By default, the nodes are returned in the order they are found in the store.

        This function sorts the nodes based on the order of the node_ids.
        """
        unsorted_nodes = self.get_nodes(collection, artifacts, node_ids, filters, limit)
        index = {node.id_: node for node in unsorted_nodes}
        sorted_nodes = [
            index[node_id] for node_id in node_ids if node_id in index  # type: ignore
        ]
        return sorted_nodes

    def get_node(
        self,
        collection: str,
        node_id: str,
    ) -> BaseNode | None:
        node = self.get_nodes(collection, node_ids=[node_id], limit=1)
        return node[0] if node else None

    def all_nodes(self, collection: str) -> list[BaseNode]:
        """Get all nodes from the store."""
        return self.filtered_nodes(collection, None)

    def filtered_nodes(
        self,
        collection: str,
        artifacts: list[str] | None = None,
        filter_dicts: list[dict[str, Any]] | None = None,
        node_ids: list[str] | None = None,
        limit: int | None = None,
        filter_condition: FilterCondition = FilterCondition.OR,
    ) -> list[BaseNode]:
        """Get nodes from the store.

        Returns a set of all nodes from the artifacts provided plus all nodes from
        the collection matching the filter dicts.

        If no filters or artifacts are provided, returns all nodes.

        Note: currently treats every filter dict as exact matches.
        """
        if not filter_dicts and not artifacts:
            # Return all nodes if no filters are provided
            return self.get_nodes(collection)

        filters = (
            MetadataFilters.from_dicts(filter_dicts, filter_condition)
            if filter_dicts
            else MetadataFilters(filters=[], condition=filter_condition)
        )

        # Get nodes from the store
        return self.get_nodes(
            collection,
            artifacts=artifacts,
            filters=filters,
            node_ids=node_ids,
            limit=limit,
        )

    def get_list_of_artifact_ids(
        self, collection: str, current_state: str = "populated"
    ) -> list[str]:
        """Get a list of all artifact ids in the collection."""
        index_store = self.index_store(collection)
        indexes = [
            index.index_id
            for index in index_store.index_structs()
            if index.index_id
            and (current_state is None or index.summary == current_state)
        ]

        def extract_artifact_id(index_id: str) -> str:
            match = re.match(r"(.*)-.*", index_id)
            if match:
                return match.group(1)
            return index_id

        artifact_ids = [extract_artifact_id(index_id) for index_id in indexes]
        return list(set(artifact_ids))

    def delete_nodes(
        self,
        collection: str,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:
        vector_store = self._vector_store_component.vector_store(collection)
        if vector_store is None:
            raise ValueError(f"Vector store for collection {collection} not found")
        if not hasattr(vector_store, "get_nodes"):
            raise ValueError(
                f"Vector store for collection {collection} does not support get_nodes"
            )
        return vector_store.delete_nodes(
            node_ids=node_ids, filters=filters, **delete_kwargs
        )

    def delete_node(self, collection: str, node_id: str) -> None:
        return self.delete_nodes(collection, node_ids=[node_id])

    def delete_filtered_nodes(
        self,
        collection: str,
        artifacts: list[str] | None = None,
        filter_dicts: list[dict[str, Any]] | None = None,
        node_ids: list[str] | None = None,
    ) -> None:
        filters = (
            MetadataFilters.from_dicts(filter_dicts, FilterCondition.OR)
            if filter_dicts
            else MetadataFilters(filters=[], condition=FilterCondition.OR)
        )

        # Create new filters for every artifact
        if artifacts:
            for artifact in artifacts:
                filters.filters.append(
                    MetadataFilter(key=MetadataKeys.ARTIFACT_ID.value, value=artifact)
                )

        return self.delete_nodes(collection, filters=filters, node_ids=node_ids)
