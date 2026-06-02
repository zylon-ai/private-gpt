import sqlalchemy
from llama_index.storage.kvstore.postgres import PostgresKVStore  # type: ignore


class PatchedPostgresKVStore(PostgresKVStore):  # type: ignore
    """Patched PostgresKVStore that escapes the table name.

    Our tables names contain "-" (they are uuids),
    which are not supported by the default PostgresKVStore.
    """

    def __init__(
        self,
        connection_string: str,
        async_connection_string: str,
        table_name: str,
        schema_name: str = "public",
        engine: sqlalchemy.engine.Engine | None = None,
        async_engine: sqlalchemy.ext.asyncio.AsyncEngine | None = None,
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = False,
    ) -> None:
        super().__init__(
            connection_string=connection_string,
            async_connection_string=async_connection_string,
            table_name=table_name,
            schema_name=schema_name,
            engine=engine,
            async_engine=async_engine,
            perform_setup=perform_setup,
            debug=debug,
            use_jsonb=use_jsonb,
        )
        self._table_class.__tablename__ = f'"{self._table_class.__tablename__}"'
