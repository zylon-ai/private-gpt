import logging

import psycopg2
import sqlalchemy
from llama_index.storage.kvstore.postgres import (  # ty:ignore[unresolved-import]
    PostgresKVStore,
)
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)


class PatchedPostgresKVStore(PostgresKVStore):
    """Patched PostgresKVStore.

    Our tables names contain "-" (they are uuids),
    which are not supported by the default PostgresKVStore.

    We also share the same connection among all KVStores to the database.
    """

    def __init__(
        self,
        session: sessionmaker[Session],
        async_session: sessionmaker[Session],
        table_name: str,
        schema_name: str = "public",
        engine: sqlalchemy.engine.Engine | None = None,
        async_engine: sqlalchemy.ext.asyncio.AsyncEngine  # ty:ignore[possibly-missing-submodule]
        | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(
            connection_string="unused",
            async_connection_string="unused",
            table_name=table_name,
            schema_name=schema_name,
            engine=engine,
            async_engine=async_engine,
            perform_setup=False,
            debug=debug,
            use_jsonb=True,
        )
        self._table_class.__tablename__ = f'"{self._table_class.__tablename__}"'
        self._session = session
        self._async_session = async_session
        self._create_schema_if_not_exists()
        self._create_tables_if_not_exists()
        self._is_initialized = True

    def _create_schema_if_not_exists(self) -> None:
        """Patched version to avoid checking before if schema exists.

        If we do in two steps, we can have problems with concurrent
        creation of the schema. Instead, we just try to create it
        """
        with self._session() as session, session.begin():
            from sqlalchemy import text

            create_schema_statement = text(
                f"CREATE SCHEMA IF NOT EXISTS {self.schema_name}"
            )
            session.execute(create_schema_statement)
            session.commit()

    def _create_tables_if_not_exists(self) -> None:
        """Patched version to avoid crash when table already exists.

        In a concurrent environment, we can have created the table
        between the check and the creation. We ignore the error
        if the table already exists.
        """
        try:
            super()._create_tables_if_not_exists()
        except sqlalchemy.exc.IntegrityError as e:  # ty:ignore[possibly-missing-submodule]
            if isinstance(e.orig, psycopg2.errors.IntegrityError):
                logger.warning(
                    "Table %s already exists, ignoring error", self.table_name
                )
                return
            raise e
        except sqlalchemy.exc.ProgrammingError as e:  # ty:ignore[possibly-missing-submodule]
            if isinstance(e.orig, psycopg2.errors.DuplicateTable):
                logger.warning(
                    "Table %s already exists, ignoring error", self.table_name
                )
                return
            raise e
