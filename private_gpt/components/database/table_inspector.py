from sqlalchemy import inspect

from private_gpt.components.database.inspected_schema import InspectedTable
from private_gpt.components.database.inspector_interface import (
    DatabaseObjectType,
    InspectedDatabaseObject,
)
from private_gpt.components.database.table_like_inspector import (
    DatabaseTableLikeInspector,
)


class DatabaseTableInspector(DatabaseTableLikeInspector):
    def get_inspector_type(self) -> str:
        return DatabaseObjectType.TABLE

    def get_objects(self, schema: str) -> list[InspectedDatabaseObject]:
        meta = inspect(self._engine)
        if not meta:
            raise ValueError("Failed to inspect the database schema.")
        tables = sorted(meta.get_table_names(schema=schema))
        result: list[InspectedDatabaseObject] = []
        for table_name in tables:
            result.append(self._extract_schema(schema, table_name, InspectedTable))
        return result
