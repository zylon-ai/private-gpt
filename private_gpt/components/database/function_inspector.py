from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from private_gpt.components.database.inspected_schema import (
    InspectedFunction,
    InspectedFunctionParams,
)
from private_gpt.components.database.inspector_interface import (
    DatabaseObjectInspector,
    DatabaseObjectType,
)


class DatabaseFunctionsInspector(DatabaseObjectInspector):
    def get_inspector_type(self) -> str:
        return DatabaseObjectType.FUNCTION

    def get_objects(self, schema: str) -> list[InspectedFunction]:
        try:
            if self._db_type in ["mssql", "microsoft"]:
                return self._get_sqlserver_functions(schema)
            if self._db_type in ["db2", "ibm_db_sa"]:
                return self._get_db2_functions(schema)

            # TODO: Postgres support is not complete
            # elif self._db_type == "postgresql":
            #    return self._get_postgresql_functions(schema)
            else:
                return []
        except SQLAlchemyError:
            return []

    def _get_postgresql_functions(self, schema: str) -> list[InspectedFunction]:
        # TODO: review this function (it is not finished)
        self._ensure_connected()
        schema_filter = f"AND n.nspname = '{schema}'" if schema else ""
        if self._is_readonly:
            schema_filter += " AND n.nspname NOT IN ('pg_temp_1', 'pg_toast_temp_1')"
            schema_filter += " AND UPPER(pg_get_functiondef(p.oid)) NOT LIKE '%DELETE%'"
            schema_filter += " AND UPPER(pg_get_functiondef(p.oid)) NOT LIKE '%UPDATE%'"
            schema_filter += " AND UPPER(pg_get_functiondef(p.oid)) NOT LIKE '%INSERT%'"
            schema_filter += " AND UPPER(pg_get_functiondef(p.oid)) NOT LIKE '%DROP%'"
            schema_filter += " AND UPPER(pg_get_functiondef(p.oid)) NOT LIKE '%ALTER%'"

        types = "('f')"  # To include only functions

        query = text(
            f"""
                SELECT
                    n.nspname as schema_name,
                    p.proname as procedure_name,
                    pg_get_function_result(p.oid) as return_type,
                    params.param_name,
                    format_type(params.param_type, NULL) as param_type_name,
                    params.param_position
                FROM pg_proc p
                JOIN pg_namespace n ON p.pronamespace = n.oid
                CROSS JOIN LATERAL (
                    SELECT *
                    FROM ROWS FROM (
                        unnest(COALESCE(p.proargnames, ARRAY_FILL(NULL::text, ARRAY[p.pronargs]))),
                        unnest(COALESCE(p.proallargtypes, p.proargtypes::oid[])),
                        generate_series(1, GREATEST(
                            array_length(COALESCE(p.proallargtypes, p.proargtypes::oid[]), 1),
                            p.pronargs
                        ))
                    ) AS t(param_name, param_type, param_position)
                    WHERE t.param_type IS NOT NULL
                ) params
                WHERE p.prokind IN {types}
                  {schema_filter}
                AND n.nspname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY n.nspname, p.proname, params.param_position
            """
        )

        functions = []
        try:
            if not self._engine:
                return []
            conn = self._engine.connect()
            result = conn.execute(query)
            current_function = None

            for row in result:
                schema, name, return_type, param_name, param_type, param_position = (
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                )
                if current_function is None or (
                    current_function.schema != schema or current_function.name != name
                ):
                    if current_function is not None:
                        functions.append(current_function)
                    current_function = InspectedFunction()
                    current_function.schema = schema
                    current_function.name = name
                    current_function.return_types = [return_type]
                    current_function.parameters = []
                if (
                    (param_name or param_position)
                    and current_function
                    and current_function.parameters
                ):
                    parameter = InspectedFunctionParams()
                    parameter.name = (
                        param_name if param_name else f"arg{param_position}"
                    )
                    parameter.data_type = param_type
                    current_function.parameters.append(parameter)

            if current_function is not None:
                functions.append(current_function)
        except Exception:
            return []
        return functions

    def _get_sqlserver_functions(self, schema: str) -> list[InspectedFunction]:

        schema_filter = f"AND s.name = '{schema}'" if schema else ""
        if self._is_readonly:
            schema_filter += "AND UPPER(m.definition) NOT LIKE '%DELETE%'"
            schema_filter += "AND UPPER(m.definition) NOT LIKE '%UPDATE%'"
            schema_filter += "AND UPPER(m.definition) NOT LIKE '%INSERT%'"
            schema_filter += "AND UPPER(m.definition) NOT LIKE '%DROP%'"
            schema_filter += "AND UPPER(m.definition) NOT LIKE '%ALTER%'"
        types = "('FN')"  # Only stored functions
        query = text(
            f"""
                SELECT
                    s.name AS schema_name,
                    o.name AS object_name,
                    CAST(ep_proc.value AS NVARCHAR(MAX)) AS comment,
                    prm.name AS parameter_name,
                    t.name AS parameter_type,
                    CAST(ep_param.value AS NVARCHAR(MAX)) AS parameter_comment
                FROM sys.objects o
                    JOIN sys.schemas s ON o.schema_id = s.schema_id
                    LEFT JOIN sys.parameters prm ON o.object_id = prm.object_id
                    LEFT JOIN sys.types t ON prm.user_type_id = t.user_type_id
                    JOIN sys.sql_modules m ON o.object_id = m.object_id
                    LEFT JOIN sys.extended_properties ep_proc ON
                        ep_proc.major_id = o.object_id
                        AND ep_proc.minor_id = 0
                        AND ep_proc.name = 'MS_Description'
                        AND ep_proc.class = 1
                    LEFT JOIN sys.extended_properties ep_param ON
                        ep_param.major_id = o.object_id
                        AND ep_param.minor_id = prm.parameter_id
                        AND ep_param.name = 'MS_Description'
                        AND ep_param.class = 2
                WHERE o.type IN {types}
                  {schema_filter}
                  ORDER BY s.name, o.name;
            """
        )

        functions = []
        try:
            self._ensure_connected()
            if not self._engine:
                return []
            conn = self._engine.connect()
            result = conn.execute(query)
            current_function = None

            for row in result:
                schema, name, comment, param_name, param_type, parameter_comment = (
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                )
                if current_function is None or (
                    current_function.schema != schema or current_function.name != name
                ):
                    if current_function is not None:
                        functions.append(current_function)
                    current_function = InspectedFunction()
                    current_function.schema = schema
                    current_function.name = name
                    current_function.comment = comment
                    current_function.parameters = []
                    current_function.return_types = []
                if param_name:
                    parameter = InspectedFunctionParams()
                    parameter.name = param_name
                    parameter.data_type = param_type
                    parameter.comment = parameter_comment
                    if not current_function.parameters:
                        current_function.parameters = []
                    current_function.parameters.append(parameter)
                elif param_type:
                    if not current_function.return_types:
                        current_function.return_types = []
                    current_function.return_types.append(param_type)

            if current_function is not None:
                functions.append(current_function)
        except Exception:
            return []
        return functions

    def _get_db2_functions(self, schema: str) -> list[InspectedFunction]:
        schema_filter = (
            f"AND LOWER(r.routineschema) = LOWER('{schema}')" if schema else ""
        )

        if self._is_readonly:
            schema_filter += " AND UPPER(r.text) NOT LIKE '%DELETE%'"
            schema_filter += " AND UPPER(r.text) NOT LIKE '%UPDATE%'"
            schema_filter += " AND UPPER(r.text) NOT LIKE '%INSERT%'"
            schema_filter += " AND UPPER(r.text) NOT LIKE '%DROP%'"
            schema_filter += " AND UPPER(r.text) NOT LIKE '%ALTER%'"

        # Only stored functions (F = Function)
        query = text(
            f"""
                SELECT
                    r.routineschema AS schema_name,
                    r.routinename AS object_name,
                    r.remarks AS comment,
                    p.parmname AS parameter_name,
                    p.typename AS parameter_type,
                    p.rowtype AS param_direction
                FROM syscat.routines r
                    LEFT JOIN syscat.routineparms p ON
                        r.routineschema = p.routineschema
                        AND r.specificname = p.specificname
                WHERE r.routinetype = 'F'
                  AND r.routineschema NOT LIKE 'SYS%'
                  {schema_filter}
                ORDER BY r.routineschema, r.routinename, p.ordinal
            """
        )

        functions = []
        try:
            self._ensure_connected()
            if not self._engine:
                return []
            conn = self._engine.connect()
            result = conn.execute(query)
            current_function = None

            for row in result:
                (schema, name, comment, param_name, param_type, param_direction,) = (
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                )

                if current_function is None or (
                    current_function.schema != schema or current_function.name != name
                ):
                    if current_function is not None:
                        functions.append(current_function)
                    current_function = InspectedFunction()
                    current_function.schema = schema
                    current_function.name = name
                    current_function.comment = comment
                    current_function.parameters = []
                    current_function.return_types = []

                if param_name and param_direction in ("P", "O", "B"):
                    # P = IN, O = OUT, B = INOUT
                    parameter = InspectedFunctionParams()
                    parameter.name = param_name
                    parameter.data_type = param_type
                    parameter.comment = None
                    assert current_function.parameters is not None
                    current_function.parameters.append(parameter)
                elif param_direction == "R" and param_type:
                    assert current_function.return_types is not None
                    current_function.return_types.append(
                        f"{param_name if param_name else''} {param_type}"
                    )

            if current_function is not None:
                functions.append(current_function)
        except Exception:
            return []
        return functions
