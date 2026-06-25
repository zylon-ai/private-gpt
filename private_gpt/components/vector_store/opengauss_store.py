"""openGauss DataVec vector store.

Native openGauss DataVec integration — does NOT go through SQLAlchemy or
PGVectorStore. Uses psycopg2 directly against the openGauss DataVec kernel
extension, which provides:

  - vector data type (up to 16000 dims)
  - distance operators:  <=>  (cosine)   <->  (L2)   <#>  (inner product)
"""

import json
import logging
import threading
from collections.abc import Sequence
from typing import Any, Literal

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from psycopg2 import sql
from psycopg2.extras import Json, RealDictCursor

from private_gpt.components.readers.nodes.utils import metadata_dict_to_tree_node

logger = logging.getLogger(__name__)

_DISTANCE_OP = {
    "cosine": "<=>",
    "l2": "<->",
    "inner_product": "<#>",
}

DEFAULT_TABLE = "embeddings"


class OpenGaussVectorStore(BasePydanticVectorStore):  # type: ignore[misc]
    """Native openGauss DataVec vector store.

    Stores each node as a row with its embedding plus a JSON-serialized copy of
    the full node so it can be reconstructed on query.
    """

    stores_text: bool = True
    flat_metadata: bool = False

    _conn: Any = PrivateAttr()
    _lock: threading.Lock = PrivateAttr()
    _schema: str = PrivateAttr()
    _table: str = PrivateAttr()
    _embed_dim: int = PrivateAttr()
    _distance: str = PrivateAttr()

    def __init__(
        self,
        connection: Any,
        schema_name: str,
        table_name: str,
        embed_dim: int,
        distance: Literal["cosine", "l2", "inner_product"] = "cosine",
    ) -> None:
        super().__init__(stores_text=True)
        self._conn = connection
        self._lock = threading.Lock()
        self._schema = schema_name
        self._table = table_name
        self._embed_dim = embed_dim
        self._distance = distance
        self._init_schema()

    # ------------------------------------------------------------------ #
    # Schema bootstrap
    # ------------------------------------------------------------------ #
    def _init_schema(self) -> None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                    sql.Identifier(self._schema)
                )
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {schema}.{table} (
                        node_id       TEXT PRIMARY KEY,
                        embedding     vector({dim}),
                        node_content  TEXT,
                        node_type     TEXT,
                        ref_doc_id    TEXT,
                        metadata      JSONB
                    )
                    """
                ).format(
                    schema=sql.Identifier(self._schema),
                    table=sql.Identifier(self._table),
                    dim=sql.Literal(self._embed_dim),
                )
            )
        self._conn.commit()

    @property
    def client(self) -> Any:
        return self._conn

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _node_to_row(node: BaseNode) -> dict[str, Any]:
        from llama_index.core.vector_stores.utils import node_to_metadata_dict

        metadata = node_to_metadata_dict(node)
        embedding = node.get_embedding()
        # openGauss DataVec vector type expects the '[v1,v2,...]' string form.
        # Passing a bare list via psycopg2 adapts it to Python repr which the
        # server cannot cast to vector; serialise explicitly instead.
        return {
            "node_id": node.node_id,
            "embedding": "[" + ",".join(str(float(x)) for x in embedding) + "]",
            "node_content": json.dumps(metadata),
            "node_type": metadata.get("_node_type", node.get_type()),
            "ref_doc_id": getattr(node, "ref_doc_id", None) or node.node_id,
            "metadata": Json(dict(node.metadata or {})),
        }

    @staticmethod
    def _row_to_node(row: dict[str, Any]) -> BaseNode:
        metadata = json.loads(row["node_content"])
        metadata["_node_type"] = row["node_type"]
        node = metadata_dict_to_tree_node(metadata)
        if row.get("embedding") is not None and node.embedding is None:
            emb = row["embedding"]
            # DataVec may return the vector as a '[v1,v2,...]' string.
            if isinstance(emb, str):
                emb = [float(x) for x in emb.strip("[]").split(",") if x]
            node.embedding = emb
        return node

    # ------------------------------------------------------------------ #
    # MetadataFilters -> SQL WHERE
    # ------------------------------------------------------------------ #
    def _filters_to_sql(
        self, filters: MetadataFilters | None, params: list[Any]
    ) -> sql.Composable | None:
        if filters is None or not filters.filters:
            return None

        def _op(f: MetadataFilter) -> sql.Composable:
            # Metadata filters operate on the JSONB `metadata` column, not on
            # top-level table columns. Use the ->> operator to extract a JSON
            # key as text and compare against the provided value.
            key = sql.SQL("metadata ->> {}").format(sql.Literal(f.key))
            if f.operator in (FilterOperator.EQ, None):
                params.append(str(f.value))
                return sql.SQL("{} = %s").format(key)
            if f.operator == FilterOperator.NE:
                params.append(str(f.value))
                return sql.SQL("{} <> %s").format(key)
            if f.operator == FilterOperator.IN:
                params.append([str(v) for v in f.value])  # type: ignore[union-attr]
                return sql.SQL("{} = ANY(%s)").format(key)
            params.append(str(f.value))
            return sql.SQL("{} = %s").format(key)

        parts: list[sql.Composable] = []
        for f in filters.filters:
            if isinstance(f, MetadataFilter):
                parts.append(_op(f))
            elif isinstance(f, MetadataFilters):
                inner = self._filters_to_sql(f, params)
                if inner is not None:
                    parts.append(sql.SQL("({})").format(inner))

        if not parts:
            return None
        joiner = (
            sql.SQL(" AND ")
            if filters.condition == FilterCondition.AND
            else sql.SQL(" OR ")
        )
        return joiner.join(parts)

    # ------------------------------------------------------------------ #
    # BasePydanticVectorStore API
    # ------------------------------------------------------------------ #
    def add(
        self,
        nodes: Sequence[BaseNode],
        **add_kwargs: Any,
    ) -> list[str]:
        if not nodes:
            return []
        rows = [self._node_to_row(n) for n in nodes]
        with self._lock, self._conn.cursor() as cur:
            cur.executemany(
                sql.SQL(
                    """
                    INSERT INTO {schema}.{table}
                        (node_id, embedding, node_content, node_type, ref_doc_id, metadata)
                    VALUES (%s, %s::vector, %s, %s, %s, %s)
                    """
                ).format(
                    schema=sql.Identifier(self._schema),
                    table=sql.Identifier(self._table),
                ),
                [
                    (
                        r["node_id"],
                        r["embedding"],
                        r["node_content"],
                        r["node_type"],
                        r["ref_doc_id"],
                        r["metadata"],
                    )
                    for r in rows
                ],
            )
        self._conn.commit()
        return [r["node_id"] for r in rows]

    async def async_add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[str]:
        return self.add(nodes, **kwargs)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                sql.SQL("DELETE FROM {schema}.{table} WHERE ref_doc_id = %s").format(
                    schema=sql.Identifier(self._schema),
                    table=sql.Identifier(self._table),
                ),
                (ref_doc_id,),
            )
        self._conn.commit()

    async def adelete(self, ref_doc_id: str, **kwargs: Any) -> None:
        self.delete(ref_doc_id, **kwargs)

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:
        params: list[Any] = []
        clauses: list[sql.Composable] = []
        if node_ids:
            clauses.append(sql.SQL("node_id = ANY(%s)"))
            params.append(list(node_ids))
        where = self._filters_to_sql(filters, params)
        if where is not None:
            clauses.append(where)
        if not clauses:
            return
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                sql.SQL("DELETE FROM {schema}.{table} WHERE {cond}").format(
                    schema=sql.Identifier(self._schema),
                    table=sql.Identifier(self._table),
                    cond=sql.SQL(" AND ").join(clauses),
                ),
                params,
            )
        self._conn.commit()

    async def adelete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **kwargs: Any,
    ) -> None:
        self.delete_nodes(node_ids=node_ids, filters=filters, **kwargs)

    def get_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        limit: int | None = None,
    ) -> list[BaseNode]:
        params: list[Any] = []
        clauses: list[sql.Composable] = []
        if node_ids:
            clauses.append(sql.SQL("node_id = ANY(%s)"))
            params.append(list(node_ids))
        where = self._filters_to_sql(filters, params)
        if where is not None:
            clauses.append(where)

        query = sql.SQL(
            "SELECT node_id, embedding, node_content, node_type "
            "FROM {schema}.{table}"
        ).format(
            schema=sql.Identifier(self._schema),
            table=sql.Identifier(self._table),
        )
        if clauses:
            query = query + sql.SQL(" WHERE ") + sql.SQL(" AND ").join(clauses)
        if limit is not None:
            query = query + sql.SQL(" LIMIT %s")
            params.append(limit)

        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return [self._row_to_node(dict(r)) for r in cur.fetchall()]

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        if query.query_embedding is None:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        top_k = query.similarity_top_k or 10
        embedding = "[" + ",".join(str(float(x)) for x in query.query_embedding) + "]"
        op = _DISTANCE_OP[self._distance]

        filter_params: list[Any] = []
        where = self._filters_to_sql(query.filters, filter_params)
        where_clause = sql.SQL(" WHERE ") + where if where is not None else sql.SQL("")

        base = sql.SQL(
            "SELECT node_id, embedding, node_content, node_type, "
            "embedding {op} %s::vector AS distance "
            "FROM {schema}.{table}{where} "
            "ORDER BY embedding {op} %s::vector LIMIT %s"
        ).format(
            op=sql.SQL(op),
            schema=sql.Identifier(self._schema),
            table=sql.Identifier(self._table),
            where=where_clause,
        )
        params: list[Any] = [embedding, *filter_params, embedding, top_k]

        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(base, params)
            rows = cur.fetchall()

        nodes: list[BaseNode] = []
        ids: list[str] = []
        sims: list[float] = []
        for r in rows:
            row = dict(r)
            node = self._row_to_node(row)
            nodes.append(node)
            ids.append(row["node_id"])
            sims.append(float(row["distance"]))
        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=sims)

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        return self.query(query, **kwargs)

    def persist(self, persist_path: str | None = None, fs: Any = None) -> None:
        self._conn.commit()
