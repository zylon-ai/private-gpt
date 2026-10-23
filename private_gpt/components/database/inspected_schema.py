import enum
from abc import abstractmethod

from sqlalchemy.engine import ObjectKind

from private_gpt.components.database.inspector_interface import (
    DatabaseObjectType,
    InspectedDatabaseObject,
)

SEPARATION = "\n"
SEPARATION_SECTION = "\n\n"


def _clean_types(type_str: str | None) -> str | None:
    if not type_str:
        return type_str
    if "(" in type_str:
        return type_str.split("(")[0]
    if "COLLATE" in type_str:
        return type_str.split("COLLATE")[0].strip()
    return type_str


### TABLE INSPECTOR COMPONENTS ###
class TableType(enum.StrEnum):
    TABLE = "TABLE"
    VIEW = "VIEW"


class InspectedForeignKey:
    referred_schema: str
    referred_table: str
    referred_columns: str
    constrained_columns: str

    def __str__(self) -> str:
        return (
            f"FK to {self.referred_schema}.{self.referred_table}.{self.referred_columns}"
            f" from ({self.constrained_columns})"
        )


class InspectedColumn:
    name: str
    type: str
    nullable: bool
    autoincrement: bool
    comment: str | None

    def __str__(self) -> str:
        null_marker = "?" if self.nullable else ""
        return f"{self.name}:{_clean_types(self.type)}{null_marker}"


class InspectedTableLike(InspectedDatabaseObject):
    schema: str
    comment: str | None
    columns: list[InspectedColumn]
    foreign_keys: list[InspectedForeignKey]
    primary_key: str | None

    @abstractmethod
    def get_type(self) -> str:
        pass

    @abstractmethod
    def _get_object_label(self) -> str:
        pass

    @classmethod
    @abstractmethod
    def get_kind(cls) -> ObjectKind:
        pass

    def __str__(self) -> str:
        cols_str = SEPARATION.join(str(col) for col in self.columns)
        fks_str = SEPARATION.join(str(fk) for fk in self.foreign_keys)

        result = f"{self._get_object_label()}: {self.schema}.{self.name}"
        if self.comment:
            result += f"{SEPARATION}-- {self.comment}"
        if len(self.columns) > 0 and cols_str:
            result += f"{SEPARATION}Columns:{SEPARATION}{cols_str}"
        if self.primary_key and len(self.primary_key) > 0:
            result += f"{SEPARATION}PK:{self.primary_key}"
        if self.foreign_keys and len(self.foreign_keys) > 0 and fks_str:
            separator = "" if len(self.foreign_keys) == 1 else SEPARATION
            result += "\nForeign Keys:"
            result += f"{separator}{fks_str}"
        return result


class InspectedTable(InspectedTableLike):
    def get_type(self) -> str:
        return DatabaseObjectType.TABLE

    def _get_object_label(self) -> str:
        return "Table"

    @classmethod
    def get_kind(cls) -> ObjectKind:
        return ObjectKind.TABLE


class InspectedView(InspectedTableLike):
    def get_type(self) -> str:
        return DatabaseObjectType.VIEW

    def _get_object_label(self) -> str:
        return "View"

    @classmethod
    def get_kind(cls) -> ObjectKind:
        return ObjectKind.VIEW


### PROCEDURE INSPECTOR COMPONENTS ###
class InspectedProcedureParams:
    name: str
    data_type: str
    all_parameters: str | None = None
    comment: str | None = None

    def __str__(self) -> str:
        comment = f"-- {self.comment}" if self.comment else ""
        if self.all_parameters:
            return self.all_parameters + comment
        return f"{self.name} {_clean_types(self.data_type)} {comment}".strip()


class InspectedProcedure(InspectedDatabaseObject):
    parameters: list[InspectedProcedureParams] | None = None
    return_types: list[str] | None = None
    comment: str | None = None

    def get_type(self) -> str:
        return DatabaseObjectType.PROCEDURE

    def __str__(self) -> str:
        result = f"Procedure: {self.schema}.{self.name}"
        if self.comment:
            result += f"-- {self.comment}{SEPARATION}"
        if self.parameters:
            result += SEPARATION

            separator = "" if len(self.parameters) == 1 else SEPARATION
            result += "Params:" + separator

            for parameter in self.parameters:
                result += str(parameter)
                result += separator

            result = result.rstrip(separator)

        if self.return_types:
            result += SEPARATION

            separator = "" if len(self.return_types) == 1 else SEPARATION
            result += "Returns:" + separator

            for ret_type in self.return_types:
                result += ret_type
                result += separator

            result = result.rstrip(separator)

        return result


class InspectedFunctionParams:
    name: str
    data_type: str
    all_parameters: str | None = None
    comment: str | None = None

    def __str__(self) -> str:
        comment = f"-- {self.comment}" if self.comment else ""
        if self.all_parameters:
            return self.all_parameters + comment
        return f"{self.name} {_clean_types(self.data_type)} {comment}".strip()


class InspectedFunction(InspectedDatabaseObject):
    parameters: list[InspectedFunctionParams] | None = None
    return_types: list[str] | None = None
    comment: str | None = None

    def get_type(self) -> str:
        return DatabaseObjectType.PROCEDURE

    def __str__(self) -> str:
        result = f"Function: {self.schema}.{self.name}"
        if self.comment:
            result += f"-- {self.comment}{SEPARATION}"
        if self.parameters:
            result += SEPARATION

            separator = "" if len(self.parameters) == 1 else SEPARATION
            result += "Params:" + separator

            for parameter in self.parameters:
                result += str(parameter)
                result += separator

            result = result.rstrip(separator)

        if self.return_types:
            result += SEPARATION

            separator = "" if len(self.return_types) == 1 else SEPARATION
            result += "Returns:" + separator

            for ret_type in self.return_types:
                result += ret_type
                result += separator

            result = result.rstrip(separator)

        return result


### SCHEMA COMPONENTS ###
class InspectedSchema:
    name: str
    _objects: list[InspectedDatabaseObject]

    def __init__(self) -> None:
        self._objects = []

    def add_object(self, obj: InspectedDatabaseObject) -> None:
        self._objects.append(obj)

    def get_objects_by_type(
        self, obj_type: DatabaseObjectType
    ) -> list[InspectedDatabaseObject]:
        return [obj for obj in self._objects if obj.get_type() == obj_type]

    @property
    def tables(self) -> list[InspectedTable]:
        return [obj for obj in self._objects if isinstance(obj, InspectedTable)]

    @property
    def views(self) -> list[InspectedView]:
        return [obj for obj in self._objects if isinstance(obj, InspectedView)]

    @property
    def procedures(self) -> list[InspectedProcedure]:
        return [obj for obj in self._objects if isinstance(obj, InspectedProcedure)]

    @property
    def functions(self) -> list[InspectedFunction]:
        return [obj for obj in self._objects if isinstance(obj, InspectedFunction)]

    @property
    def all_objects(self) -> list[InspectedDatabaseObject]:
        return self._objects

    def __str__(self) -> str:
        return f"Schema: {self.name}{SEPARATION}" + SEPARATION_SECTION.join(
            str(obj) for obj in self._objects
        )
