from sqlalchemy import inspect

from private_gpt.components.database.inspected_schema import InspectedView
from private_gpt.components.database.inspector_interface import (
    DatabaseObjectType,
    InspectedDatabaseObject,
)
from private_gpt.components.database.table_like_inspector import (
    DatabaseTableLikeInspector,
)


class DatabaseViewInspector(DatabaseTableLikeInspector):
    def get_inspector_type(self) -> str:
        return DatabaseObjectType.VIEW

    def get_objects(self, schema: str) -> list[InspectedDatabaseObject]:
        meta = inspect(self._engine)
        if not meta:
            raise ValueError("Failed to inspect the database schema.")
        views = sorted(meta.get_view_names(schema=schema))
        result: list[InspectedDatabaseObject] = []

        for view_name in views:
            result.append(self._extract_schema(schema, view_name, InspectedView))

        return result
