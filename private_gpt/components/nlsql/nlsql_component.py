import logging
from typing import Any
from urllib.parse import quote_plus

from injector import inject, singleton
from llama_index import ServiceContext, SQLDatabase, VectorStoreIndex
from llama_index.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.objects import ObjectIndex, SQLTableNodeMapping, SQLTableSchema

from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class NLSQLComponent:
    sqlalchemy_engine: Any
    sql_database: Any
    metadata_obj: Any

    @inject
    def __init__(self, settings: Settings) -> None:
        if settings.context_database.enabled:
            dialect = settings.context_database.db_dialect
            driver = settings.context_database.db_driver
            host = settings.context_database.db_host
            user = settings.context_database.db_user
            password = settings.context_database.db_password
            database = settings.context_database.database
            tables = settings.context_database.tables
            try:
                from sqlalchemy import (
                    MetaData,
                )
                from sqlalchemy.engine import create_engine

                engine = create_engine(
                    f"{dialect}+{driver}://{user}:%s@{host}/{database}"
                    % quote_plus(password)
                )
            except BaseException as error:
                raise ValueError(
                    f"Unable to initialise connection to SQL Database\n{error}"
                ) from error

            metadata_obj = MetaData()
            metadata_obj.reflect(engine)
            sql_database = SQLDatabase(engine, include_tables=tables)
            self.sqlalchemy_engine = engine
            self.sql_database = sql_database
            self.metadata_obj = metadata_obj

    def get_nlsql_query_engine(
        self,
        service_context: ServiceContext,
    ) -> SQLTableRetrieverQueryEngine:
        table_node_mapping = SQLTableNodeMapping(self.sql_database)
        table_schema_objs = []
        for table_name in self.metadata_obj.tables:
            table_schema_objs.append(SQLTableSchema(table_name=table_name))
        obj_index = ObjectIndex.from_objects(
            table_schema_objs,
            table_node_mapping,
            VectorStoreIndex,
            service_context=service_context,
        )
        return SQLTableRetrieverQueryEngine(
            service_context=service_context,
            sql_database=self.sql_database,
            table_retriever=obj_index.as_retriever(similarity_top_k=1),
        )
