from abc import ABC, abstractmethod
from typing import Any

from sqlalchemy import inspect, text

from private_gpt.components.database.inspected_schema import (
    InspectedColumn,
    InspectedForeignKey,
    InspectedTableLike,
)
from private_gpt.components.database.inspector_interface import (
    DatabaseObjectInspector,
    InspectedDatabaseObject,
)


class DatabaseTableLikeInspector(DatabaseObjectInspector, ABC):
    @abstractmethod
    def get_objects(self, schema: str) -> list[InspectedDatabaseObject]:
        pass

    @abstractmethod
    def get_inspector_type(self) -> str:
        pass

    def _extract_schema(
        self, schema: str, table_name: str, obj_class: type[InspectedTableLike]
    ) -> InspectedTableLike:
        self._ensure_connected()
        meta = inspect(self._engine)

        if not meta:
            raise ValueError("Failed to inspect the database schema.")

        table_key = (schema, table_name)

        # Get columns
        multi_cols = meta.get_multi_columns(
            schema=schema, filter_names=[table_name], kind=obj_class.get_kind()
        )
        cols = multi_cols.get(table_key, [])

        # Get primary key
        multi_pks = meta.get_multi_pk_constraint(
            schema=schema, filter_names=[table_name], kind=obj_class.get_kind()
        )
        pk = multi_pks.get(table_key, {})

        # Get foreign keys
        multi_fks = meta.get_multi_foreign_keys(
            schema=schema, filter_names=[table_name], kind=obj_class.get_kind()
        )
        fks = multi_fks.get(table_key, [])

        # Process columns
        col_comments = self._get_column_comments(schema, table_name)
        columns_out: list[InspectedColumn] = []
        for c in cols:
            new_col = InspectedColumn()
            new_col.name = c.get("name")
            new_col.type = self._safe_str(c.get("type"))  # type: ignore
            new_col.nullable = bool(c.get("nullable", True))
            new_col.comment = col_comments.get(c.get("name"))
            columns_out.append(new_col)

        fks_out = []
        for fk in fks:
            fk_out = InspectedForeignKey()
            fk_out.referred_schema = fk.get("referred_schema")
            fk_out.referred_table = fk.get("referred_table")
            fk_out.referred_columns = ",".join(fk.get("referred_columns") or [])
            fk_out.constrained_columns = ",".join(fk.get("constrained_columns") or [])
            fks_out.append(fk_out)

        table_comment = self._get_table_comment(schema, table_name)

        out = obj_class()
        out.schema = schema
        out.name = table_name
        out.comment = table_comment
        out.columns = columns_out
        out.primary_key = ",".join(pk.get("constrained_columns") or [])
        out.foreign_keys = fks_out

        return out

    def _get_table_comment(self, schema: str, table: str) -> str | None:
        if not self._engine:
            return None
        conn = self._engine.connect()
        meta = inspect(self._engine)  # type: ignore
        try:
            data = meta.get_table_comment(table_name=table, schema=schema)  # type: ignore
            if data and isinstance(data, dict):
                return data.get("text") or None
        except Exception:
            return None

        # Fallback query
        q = text(
            """
            SELECT obj_description((quote_ident(:schema) || '.' || quote_ident(:table))::regclass, 'pg_class')
            """
        )
        # noinspection PyBroadException
        try:
            res = conn.execute(q, {"schema": schema, "table": table}).scalar()
            return res if res else None
        except Exception:
            return None

    def _get_column_comments(self, schema: str, table: str) -> dict[str, str | None]:
        """Returns {column_name: comment or None}."""
        conn = self._ensure_connected()
        q = text(
            """
            SELECT a.attname                        AS column_name,
                   col_description(c.oid, a.attnum) AS comment
            FROM pg_attribute a
                     JOIN pg_class c ON a.attrelid = c.oid
                     JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = :schema
              AND c.relname = :table
              AND a.attnum > 0
              AND NOT a.attisdropped
            """
        )
        comments: dict[str, str | None] = {}
        try:
            for row in conn.execute(q, {"schema": schema, "table": table}):
                comments[row.column_name] = row.comment
        except Exception:
            pass
        return comments

    def _safe_str(self, val: Any) -> str | None:
        if val is None:
            return None
        # noinspection PyBroadException
        try:
            return str(val)
        except Exception:
            return repr(val)
