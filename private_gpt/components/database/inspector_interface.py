import enum
from abc import ABC, abstractmethod
from collections.abc import Sequence

from sqlalchemy import Connection, Engine

from private_gpt.components.database.connection_factory import (
    DatabaseDialect,
    classify_dialect,
    create_engine_for_connection_string,
)


class DatabaseObjectType(enum.StrEnum):
    TABLE = "TABLE"
    VIEW = "VIEW"
    PROCEDURE = "PROCEDURE"
    FUNCTION = "FUNCTION"


class InspectedDatabaseObject(ABC):
    schema: str
    name: str

    @abstractmethod
    def get_type(self) -> str:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class DatabaseObjectInspector(ABC):
    _engine: Engine | None
    _db_type: str
    _is_readonly: bool
    _connection: Connection | None = None
    connection_string: str
    dialect: DatabaseDialect

    def __init__(
        self,
        engine: Engine | None,
        connection: Connection | None,
        connection_string: str,
        is_readonly: bool = True,
    ):
        self._engine = engine
        self._db_type = engine.dialect.name.lower() if engine else "unknown"
        self.dialect = classify_dialect(self._db_type)
        self._is_readonly = is_readonly
        self.connection_string = connection_string
        self._connection = connection

    @abstractmethod
    def get_objects(self, schema: str) -> Sequence[InspectedDatabaseObject]:
        pass

    @abstractmethod
    def get_inspector_type(self) -> str:
        pass

    def _ensure_connected(self) -> Connection:
        if self._connection and not self._connection.closed:
            return self._connection

        self._engine = create_engine_for_connection_string(
            self.connection_string
            # TODO: SSL Certificates
        )
        conn = self._engine.connect()
        self._connection = conn
        return conn
