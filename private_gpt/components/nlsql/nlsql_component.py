import logging
from urllib.parse import quote_plus

from injector import inject, singleton
from llama_index import ServiceContext, SQLDatabase, VectorStoreIndex
from llama_index.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.objects import ObjectIndex, SQLTableNodeMapping, SQLTableSchema
from sqlalchemy import (
    MetaData,
)
from sqlalchemy.engine import create_engine
from sqlalchemy.engine.base import Engine

from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class NLSQLComponent:
    sqlalchemy_engine: Engine
    sql_database: SQLDatabase
    metadata_obj: MetaData

    @inject
    def __init__(self, settings: Settings) -> None:
        dialect = settings.sqldatabase.dialect
        driver = settings.sqldatabase.driver
        host = settings.sqldatabase.host
        user = settings.sqldatabase.user
        password = settings.sqldatabase.password
        database = settings.sqldatabase.database
        tables = settings.sqldatabase.tables

        engine = create_engine(
            f"{dialect}+{driver}://{user}:%s@{host}/{database}" % quote_plus(password)
        )
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
        for table_name in self.metadata_obj.tables.keys():
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
