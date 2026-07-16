import asyncio
import contextlib
import enum
import functools
import logging
import pickle
import re
import time
from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from Levenshtein import distance
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from pandas import DataFrame
from pydantic import BaseModel, Field
from sqlalchemy import Connection, Engine, create_engine, inspect, text
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError

from private_gpt.components.cache import Cache, MemoryCache
from private_gpt.components.chat.models.chat_config_models import (
    CondensationConfig,
    ResolvedChatRequest,
    ResolvedSystemConfig,
)
from private_gpt.components.chat.processors.chat_history.memory.utils.content import (
    messages_to_history_str,
)
from private_gpt.components.database.function_inspector import (
    DatabaseFunctionsInspector,
)
from private_gpt.components.database.inspected_schema import InspectedSchema
from private_gpt.components.database.inspector_interface import (
    DatabaseObjectInspector,
    InspectedDatabaseObject,
)
from private_gpt.components.database.procedure_inspector import (
    DatabaseProcedureInspector,
)
from private_gpt.components.database.table_inspector import DatabaseTableInspector
from private_gpt.components.database.view_inspector import DatabaseViewInspector
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.di import get_global_injector
from private_gpt.events.models import TextBlock
from private_gpt.utils.dependencies import format_missing_dependency_message

if TYPE_CHECKING:
    from sqlalchemy import Row

logger = logging.getLogger(__name__)

try:
    import sqlglot  # type: ignore[import-not-found]
    from sqlglot import Dialects  # type: ignore[import-not-found]
    from sqlglot.errors import ParseError  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError) as e:
    raise ImportError(
        format_missing_dependency_message(
            "Database query",
            extras=(
                "database-postgres",
                "database-mysql",
                "database-mssql",
                "database-db2",
                "database",
            ),
        )
    ) from e


@functools.cache
def _load_ibm_db() -> Any:
    try:
        import ibm_db  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            format_missing_dependency_message(
                "DB2 database query",
                extras=("database-db2", "database"),
            )
        ) from e

    return ibm_db


class ErrorType(enum.StrEnum):
    UNKNOWN = "UNKNOWN"
    NOT_SUPPORTED_TYPE_MSSQL = "HY106"  # Error code for MSSQL unsupported type
    RETURNED_SQL_CURSOR = (
        "RETURNED_SQL_CURSOR"  # Postgres procedure returned a SQL cursor
    )
    INVALID_PARAM_MODE_DB2 = (
        "42886"  # DB2 procedure invalid parameter mode (e.g., OUT param used as IN)
    )


class ErrorQueryResult(BaseModel):
    description: str = Field(description="Error message if an error occurred.")
    type: ErrorType = Field(description="The type of error.")

    def __init__(self, description: str, type: ErrorType = ErrorType.UNKNOWN):
        super().__init__(description=description, type=type)

    def __str__(self) -> str:
        if self.type == ErrorType.NOT_SUPPORTED_TYPE_MSSQL:
            return (
                f"{self.description}\n\n"
                "CRITICAL - How to fix:\n"
                "1. Identify the column at position 'column-index' (starting from 0) in your SELECT clause\n"
                "2. That column has an unsupported spatial type (marked as 'NULL' in schema)\n"
                "3. Add .ToString() ONLY: columnName.ToString() AS columnName\n"
                "4. DO NOT use ISNULL(), CAST(), CONVERT(), or any other SQL functions\n"
                "6. If using SELECT *, replace with explicit column names first\n"
                "7. Keep all other parts unchanged (WHERE, JOIN, ORDER BY)\n"
                "Example fix: c.DeliveryLocation.ToString() AS DeliveryLocation\n"
            )
        return self.description


class QueryResult(BaseModel):
    query: str | None = Field(
        default=None, description="The SQL query that was executed."
    )
    rows_text: str | None = Field(
        default=None, description="The result of the query or an error message."
    )
    columns: list[str] | None = Field(
        default=None, description="The column names of the result set."
    )
    rows: Sequence[Any] = Field(
        default_factory=list, description="The raw rows returned by the query."
    )
    error: ErrorQueryResult | None = Field(
        default=None, description="Error message if an error occurred."
    )
    row_count: int = Field(
        description="The number of rows returned by the query, -1 if an error occurred.",
    )
    warning: str | None = Field(
        default=None,
        description="Warning message if the query executed but with potential issues (e.g., timeout, partial results).",
    )
    _csv_cache: str | None = None

    def as_csv(self) -> str:
        if self._csv_cache is not None:
            return self._csv_cache
        if self.error:
            return f"Error executing query: {self.error}"
        if not self.rows or not self.columns:
            return "No results found."

        table_str = DataFrame(
            columns=self.columns,
            data=self.rows,
        ).to_csv(index=False)

        if not table_str:
            return "No results found."

        self._csv_cache = table_str
        return table_str


class DatabaseResultEvent(BaseModel):
    content: list[Any] = Field(
        default_factory=list, description="The content blocks from the analysis."
    )
    is_error: bool = Field(
        False,
        description="Indicates if the analysis resulted in an error.",
    )


_DEFAULT_EXCLUDED_SCHEMAS = {
    # Postgres specific
    "pg_catalog",
    "information_schema",
    "pg_toast",
    "pg_temp_1",
    "pg_toast_temp_1",
    # MSSQL specific
    "db_accessadmin",
    "db_backupoperator",
    "db_datareader",
    "db_datawriter",
    "db_ddladmin",
    "db_denydatareader",
    "db_denydatawriter",
    "db_owner",
    "db_securityadmin",
    "PowerBI",
    "guest",
    "sys",
    "nullid",
    "sqlj",
}


class InspectorConfig(BaseModel):
    tables: bool = Field(default=True)
    views: bool = Field(default=True)
    procedures: bool = Field(default=True)
    functions: bool = Field(default=True)


_ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class DatabaseQueryGenerator:
    """Helper to handle natural language queries against a SQL database.

    Generally PandasAI is used for tabular data analysis, but it simply
    doesn't work well for relational databases. This service uses an LLM
    to convert natural language queries into SQL, executes them, and returns
    the results.
    """

    connection_string: str
    ssl: bool
    schemas: list[str] | None
    max_retries: int
    is_readonly: bool
    _engine: Engine | None
    _connection: Connection | None
    _dialect: str | None
    inspector_config: InspectorConfig
    batch_size: int
    timeout_seconds: int | None
    max_mb_result: int | None = 100

    def __init__(
        self,
        connection_string: str,
        schemas: list[str] | None = None,
        ssl: bool = False,
        max_retries: int = 3,
        is_readonly: bool = True,
        enable_tables: bool = True,
        enable_views: bool = True,
        enable_procedures: bool = True,
        enable_functions: bool = True,
        description: str = "",
        batch_size: int = 1000,
        timeout_seconds: int | None = None,
        max_mb_result: int | None = None,
        cache: Cache | None = None,
    ):
        # Need to do it lazily to avoid circular dependency
        self.connection_string = connection_string
        self.ssl = ssl
        self.schemas = schemas
        self.max_retries = max_retries
        self.is_readonly = is_readonly
        self._connection = None
        self._engine = None
        self.inspector_config = InspectorConfig()
        self.inspector_config.views = enable_views
        self.inspector_config.tables = enable_tables
        self.inspector_config.procedures = enable_procedures
        self.inspector_config.functions = enable_functions
        self.description = description
        self._dialect = self._extract_dialect()
        self._prepare_connection_string()
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.max_mb_result = max_mb_result
        self.cache = cache or MemoryCache(max_entries=1000)

    def _extract_database_name(self) -> str:
        # crude way to extract the database name from the connection string
        # works for postgres style connection strings
        # postgresql://user:password@host:port/dbname
        try:
            # extract the part after the last /
            db_name = self.connection_string.rsplit("/", 1)[-1]
            # remove any query parameters
            db_name = db_name.split("?", 1)[0]
            return db_name if db_name else "unknown"
        except Exception:
            return "unknown"

    def _extract_dialect(self) -> str | None:
        try:
            protocol = self.connection_string.split("://", 1)[0]
            if "mssql" in protocol:
                # if we don't do this, distance will pick "mysql"
                # over "tsql" due to shorter length
                return cast(str, Dialects.TSQL.value)

            best_distance = -1
            best_dialect: str | None = None
            for dialect in Dialects:
                dist = distance(protocol.lower(), dialect.value.lower())
                if best_distance == -1 or dist < best_distance:
                    best_distance = dist
                    best_dialect = cast(str, dialect.value)
            return best_dialect
        except Exception:
            return None

    def _is_readonly_query(self, sql: str) -> bool:
        with contextlib.suppress(Exception):
            parsed = sqlglot.parse_one(sql, dialect=self._dialect)
            return parsed.find(sqlglot.expressions.Select) is not None and not any(
                parsed.find(expr)
                for expr in [
                    sqlglot.expressions.Insert,
                    sqlglot.expressions.Update,
                    sqlglot.expressions.Delete,
                    sqlglot.expressions.Drop,
                    sqlglot.expressions.Create,
                    sqlglot.expressions.Alter,
                ]
            )

        return False  # If parsing fails, assume unsafe

    def _create_transaction_template(self) -> str:
        """Generate transaction template using SQLGlot transpilation."""
        if not self.is_readonly:
            return "%s;"

        template = []
        if self._dialect in [Dialects.POSTGRES]:
            template = ["SET TRANSACTION READ ONLY"]
        elif self._dialect in [Dialects.TSQL]:
            # Whitelisted fallback
            template = ["BEGIN", "ROLLBACK"]
            template = [
                sqlglot.transpile(part, write=self._dialect)[0] for part in template
            ]

        if not template:
            return "%s;"

        prefix = f"{template[0]};\n"
        suffix = f";\n{template[-1]};" if template[-1] != template[0] else ""
        return f"{prefix}%s{suffix}"

    def _create_safe_query_block(self, queries: list[str]) -> str:
        """Create safe query block."""
        template = self._create_transaction_template()
        return template % ";\n".join(queries)

    def _fix_cursor_result(self, query: str) -> str:
        # Currently only support Postgres cursors, but we need to check with order
        # type of returns in the postgres procedures
        if self._dialect in [Dialects.POSTGRES]:
            call_pattern = r"CALL\s+([\w\.]+)\s*\((.*?)\)"
            match = re.search(call_pattern, query, re.IGNORECASE | re.DOTALL)

            if not match:
                return query

            proc_name = match.group(1)
            params_str = match.group(2).strip()

            # Sometimes when the user does more than one call
            # the model generate correctly the call with cursor
            if re.search(r"llm_cursor_\d+", params_str):
                return query

            param_values = []

            if ":=" in params_str:
                param_pattern = r"(\w+)\s*:=\s*([^,\)]+)"
                for param_match in re.finditer(param_pattern, params_str):
                    value = param_match.group(2).strip()
                    param_values.append(value)
            else:
                param_pattern = r"'[^']*'|\"[^\"]*\"|\b[^,]+\b"
                param_values = [
                    v.strip()
                    for v in re.findall(param_pattern, params_str)
                    if v.strip()
                ]

            params_clean = ", ".join(param_values)
            cursor_name = f"llm_cursor_{hash(proc_name + params_clean) & 0xFFFFFF}"

            if params_clean:
                params_with_cursor = f"{params_clean}, '{cursor_name}'"
            else:
                params_with_cursor = f"'{cursor_name}'"

            modified_query = f'CALL {proc_name}({params_with_cursor}); FETCH ALL FROM "{cursor_name}";'

            return modified_query

        return query

    def _prepare_connection_string(self) -> None:
        parsed = urlparse(self.connection_string)
        params = parse_qs(parsed.query, keep_blank_values=True)

        if self._dialect in [Dialects.TSQL]:
            # MSSQL specific adjustments
            if not any(k.lower() == "encrypt" for k in params):
                params["Encrypt"] = ["yes" if self.ssl else "no"]

            if not any(k.lower() == "driver" for k in params):
                if not parsed.scheme.startswith("mssql+pyodbc"):
                    raise ValueError(
                        "MSSQL connection requires pyodbc scheme when no driver specified"
                    )
                params["driver"] = ["ODBC Driver 18 for SQL Server"]

        elif self._dialect in [Dialects.MYSQL]:
            # MySQL specific adjustments
            if parsed.scheme == "mysql" or parsed.scheme == "mysql+mysqldb":
                parsed = parsed._replace(scheme="mysql+pymysql")
            elif not parsed.scheme.startswith("mysql+pymysql"):
                raise ValueError(
                    "MySQL connection requires mysql, mysql+mysqldb, or mysql+pymysql scheme"
                )
            if not any(k.lower() == "charset" for k in params):
                params["charset"] = ["utf8mb4"]

        flattened = {k: v[0] if len(v) == 1 else v for k, v in params.items()}
        new_parsed = parsed._replace(query=urlencode(flattened, doseq=True))
        self.connection_string = str(urlunparse(new_parsed))

    def _check_connection(self) -> str | None:
        """Check if the database connection can be established.

        Returns None if successful, or an error message if failed.
        """
        try:
            conn = self._ensure_connected()
            match self._engine.dialect.name.lower() if self._engine else "unknown":
                case "db2" | "ibm_db_sa":
                    conn.execute(text("SELECT 1 FROM SYSIBM.SYSDUMMY1"))
                case _:
                    conn.execute(text("SELECT 1"))
            return None
        except (SQLAlchemyError, ProgrammingError) as e:
            logging.error(e)
            return f"Failed to connect to database '{self._extract_database_name()}'"
        except Exception as e:
            logging.error(e)
            return f"Failed to connect to database '{self._extract_database_name()}'"
        finally:
            self._disconnect()

    async def check_connection(self) -> str | None:
        result = await asyncio.to_thread(self._check_connection)
        if result is None:
            # Successful connection
            # Pre-warm the schema cache asynchronously
            def _extract_schema_warmup() -> None:
                try:
                    self._extract_database_schema()
                except Exception:
                    logger.warning(
                        f"Failed to pre-warm schema cache for database '{self._extract_database_name()}'"
                    )
                finally:
                    self._disconnect()

            asyncio.create_task(  # noqa: RUF006
                asyncio.to_thread(_extract_schema_warmup)
            )
        return result

    def _query(self, query: str, execute_batch: bool = False) -> QueryResult:
        def execute_query(q: str) -> QueryResult:
            safe_query = self._create_safe_query_block([q])
            conn = self._ensure_connected()
            cursor = conn.execute(text(safe_query))

            rows = cursor.fetchall()
            column_names = cursor.keys()

            return QueryResult(
                query=q,
                rows=rows,
                columns=[str(s) for s in column_names],
                row_count=len(rows),
            )

        def execute_query_batched(q: str) -> QueryResult:
            start_time = time.perf_counter()
            all_rows: list[Row[Any]] = []

            safe_query = self._create_safe_query_block([q])
            conn = self._ensure_connected()

            result = conn.execution_options(stream_results=True).execute(
                text(safe_query)
            )

            # result.keys() is needed here because result is a SQLAlchemy
            # Result object, not a dict
            columns = [str(s) for s in result.keys()]  # noqa: SIM118

            count = 0
            timeout_reached = False

            batches_limit = None
            # Máximo de 100 MB
            max_bytes = self.max_mb_result * 1024 * 1024 if self.max_mb_result else None
            size_limit_reached = False

            while True:
                if (
                    self.timeout_seconds
                    and time.perf_counter() - start_time >= self.timeout_seconds
                ):
                    timeout_reached = True
                    break

                batch = result.fetchmany(self.batch_size)
                count += 1

                if not batch:
                    break

                all_rows.extend(batch)

                if count == 1 and max_bytes:
                    estimated_bytes_per_batch = len(
                        pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)
                    )
                    if estimated_bytes_per_batch > 0:
                        batches_limit = (
                            int(max_bytes * 0.9)
                        ) // estimated_bytes_per_batch

                if batches_limit and count > batches_limit:
                    size_limit_reached = True
                    break

            result.close()

            warning = None

            if size_limit_reached:
                warning = (
                    f"Query result size limit of {self.max_mb_result} MB reached after fetching {len(all_rows)}"
                    f" rows. The result set may be incomplete due to size constraints."
                )

            if timeout_reached:
                warning = (
                    f"Query time limit reached after fetching {len(all_rows)} rows. The result set may be "
                    f"incomplete due to time constraints."
                )

            return QueryResult(
                query=q,
                rows=all_rows,
                columns=columns,
                row_count=len(all_rows),
                warning=warning if warning else None,
            )

        exec_func = execute_query_batched if execute_batch else execute_query

        original_sql: str = ""
        result: QueryResult | None = None
        # 1. Try to run the query as-is
        try:
            original_sql = self._extract_sql_code(query, transpile_sql=False)
            result = exec_func(original_sql)
            # TODO: review for add cursor support
            # result = self._check_result(result)
            if result and not result.error:
                return result
        except (SQLAlchemyError, ProgrammingError) as e:
            error_str = str(e).upper()
            # Determine error type based on SQLSTATE or error code
            if ErrorType.NOT_SUPPORTED_TYPE_MSSQL in error_str:
                error_type = ErrorType.NOT_SUPPORTED_TYPE_MSSQL
            elif f"SQLSTATE={ErrorType.INVALID_PARAM_MODE_DB2}" in error_str or (
                "PARAMS BOUND NOT MATCHING" in error_str and "REQUIRED" in error_str
            ):
                error_type = ErrorType.INVALID_PARAM_MODE_DB2
            else:
                error_type = ErrorType.UNKNOWN
            result = QueryResult(
                query=query,
                error=ErrorQueryResult(description=str(e), type=error_type),
                row_count=-1,
            )

        # 2. Try to transpile and run again
        with contextlib.suppress(Exception):
            transpiled_sql = self._extract_sql_code(query, transpile_sql=True)
            if transpiled_sql != original_sql:
                result = exec_func(transpiled_sql)
                if result:
                    return result

        # 3. Try to fix NOT_SUPPORTED_TYPE errors
        if (
            result
            and result.error
            and result.error.type == ErrorType.NOT_SUPPORTED_TYPE_MSSQL
        ):
            with contextlib.suppress(Exception):
                fixed_sql = self._try_to_fix_unsupported_type_error(original_sql)
                if fixed_sql and fixed_sql != original_sql:
                    result = exec_func(fixed_sql)
                    if result:
                        return result

        # 4. Try to fix DB2 procedure OUT parameter errors
        if (
            result
            and result.error
            and result.error.type == ErrorType.INVALID_PARAM_MODE_DB2
        ):
            with contextlib.suppress(Exception):
                fixed_result = self._execute_db2_procedure(original_sql)
                if fixed_result and not fixed_result.error:
                    return fixed_result

        # TODO: This code is for postgres support but now is not used
        # 4. Try to fix returned SQL cursor errors
        # if (
        #    result
        #    and result.error
        #    and result.error.type == ErrorType.RETURNED_SQL_CURSOR
        # ):
        #    with contextlib.suppress(Exception):
        #        fixed_sql = self._fix_cursor_result(original_sql)
        #        if fixed_sql and fixed_sql != original_sql:
        #            result = exec_func(fixed_sql)
        #            if result:
        #                return result

        return (
            result
            if result
            else QueryResult(
                query=query,
                error=ErrorQueryResult("Failed to execute query for unknown reasons."),
                row_count=-1,
            )
        )

    def _check_batched_support(self) -> bool:
        # Change to with list (Now support mssql and db2 and works with bacthed)
        # But postgres doesn't support batch with procedures that return cursors,
        unsupported_dialects = [Dialects.POSTGRES]
        return self._dialect not in unsupported_dialects

    async def _query_batched_stream(self, query: str) -> QueryResult:
        return await asyncio.to_thread(
            self._query, query, execute_batch=self._check_batched_support()
        )

    async def query(
        self,
        query: str,
        additional_context: str | None = None,
    ) -> QueryResult:
        last_error: ErrorQueryResult | None = None
        last_generated_query: str | None = None
        try:
            extracted_schemas = await asyncio.to_thread(self._extract_database_schema)
        except SQLAlchemyError as e:
            return QueryResult(
                error=ErrorQueryResult(f"Failed to extract database schema: {e!s}"),
                row_count=-1,
            )

        if not extracted_schemas:
            return QueryResult(
                error=ErrorQueryResult("No database schema found."),
                row_count=-1,
            )

        for _i in range(self.max_retries):
            sql_query = await self.generate_sql_query(
                natural_language_query=query,
                schemas=extracted_schemas,
                additional_context=additional_context,
                last_generated_query=last_generated_query,
                last_error=last_error,
            )
            if "NO_QUERY" in sql_query.strip().upper():
                return QueryResult(
                    query=None,
                    rows_text="No relevant tables found to answer the question.",
                    row_count=0,
                )
            last_generated_query = sql_query

            result = await self._query_batched_stream(sql_query)
            if result and not result.error:
                return result

            last_error = result.error if result else ErrorQueryResult("Unknown error")

        return QueryResult(
            query=last_generated_query,
            error=last_error,
            row_count=-1,
        )

    def _ensure_connected(self) -> Connection:
        if self._connection and not self._connection.closed:
            return self._connection

        self._engine = create_engine(
            self.connection_string
            # TODO: SSL Certificates
        )
        conn = self._engine.connect()
        self._connection = conn
        return conn

    def _disconnect(self) -> None:
        if self._connection and not self._connection.closed:
            self._connection.close()
            self._connection = None
        if self._engine:
            self._engine.dispose()
            self._engine = None

    def _get_examples_by_dialect(self) -> str:
        value = self._engine.dialect.name.lower() if self._engine else self._dialect
        system_prompt = f"The database SQL dialect is: {value}."

        match value:
            case "db2" | "ibm_db_sa":
                system_prompt += (
                    " For DB2 procedures: input parameters (Params) use literal values, output parameters (Returns) use '?' placeholders."
                    " Example with 1 input + 3 outputs: CALL schema.procedure_name('value1', ?, ?, ?);"
                )
        return system_prompt

    async def generate_sql_query(
        self,
        natural_language_query: str,
        schemas: list[InspectedSchema],
        additional_context: str | None = None,
        last_generated_query: str | None = None,
        last_error: ErrorQueryResult | None = None,
    ) -> str:
        # Need to do it lazily to avoid circular dependency
        from private_gpt.server.chat.chat_service import ChatService

        chat_service = get_global_injector().get(ChatService)

        system_prompt = "Given the following database schema information:\n\n"
        system_prompt += "\n\n".join(str(schema) for schema in schemas).strip()
        system_prompt += "\n\n"
        if self.description:
            system_prompt += f"Schema description: {self.description}\n\n"
        system_prompt += (
            "Only generate the SQL query, do not include any explanations. "
        )

        system_prompt += self._get_examples_by_dialect()

        system_prompt += (
            "Do not generate style characters, the output SQL will be executed as-is, "
            "it will not be displayed in a terminal. "
        )
        system_prompt += "Prefer standard SQL syntax over database-specific syntax. "
        system_prompt += (
            "Use views or procedures instead of underlying tables whenever possible. "
        )
        system_prompt += "If there is no relevant tables that can answer the question. "
        system_prompt += "Return replying 'NO_QUERY'. "
        system_prompt += "When asked about data that mostly contains IDs or non-informative data (e.g. boolean, timestamps, UUIDs) "
        system_prompt += "attempt to add a relevant column that is human-readable (eg. name, title...) "
        system_prompt += "to the query unless explicitly told not to. "
        system_prompt += "The output MUST be valid SQL, no other text or explanations. "

        example = (
            "(e.g., TableName.ColumnName)"
            if self._dialect in [Dialects.TSQL.value]
            else ""
        )

        system_prompt += f"Before referencing any column {example}, verify that column exists in that specific table's column list. "
        system_prompt += "If a column does not exist in Table A but exists in related Table B, you MUST JOIN Table B first to access it. "
        system_prompt += "If no direct foreign key exists between two tables, use intermediate tables to connect them. "
        system_prompt += "Never assume column locations or relationships not explicitly documented in the schema. "

        if last_generated_query and last_error:
            system_prompt += f"\n Previous attempt was:\n{last_generated_query}\n"
            system_prompt += "But it resulted in an error:\n"
            system_prompt += f"{last_error}\n"
            system_prompt += "Please correct the SQL query."

        user_prompt = f"Generate an SQL query for the following request: {natural_language_query}\n"
        if additional_context:
            user_prompt += f"Additional context: {additional_context}\n"

        messages: list[ChatMessage] = [
            ChatMessage(
                role="user",
                content=user_prompt,
            )
        ]

        llm_component = get_global_injector().get(LLMComponent)

        tokenizer = llm_component.tokenizer
        max_model_tokens = llm_component.metadata().context_window

        final_history: list[ChatMessage] = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=system_prompt,
            ),
            *messages,
        ]
        chat_history = await asyncio.to_thread(messages_to_history_str, final_history)
        available_tokens = max_model_tokens - (max_model_tokens // 5)  # 20% buffer
        if tokenizer is not None:
            current_tokens = len(tokenizer(chat_history))
            available_tokens -= current_tokens

        sampling_params: dict[str, Any] = {}
        if available_tokens <= 0:
            # TODO: TLDR strategy don't work well here,
            #  need to implement a especially TLDR for schema
            raise ValueError("The database schema is too long to fit in the model.")
        if available_tokens > 0:
            sampling_params["max_tokens"] = available_tokens

        response = await chat_service.chat(
            ResolvedChatRequest(
                messages=messages,
                system=ResolvedSystemConfig(
                    prompt=system_prompt, use_default_prompt=False
                ),
                condensation=CondensationConfig(enabled=False),
                sampling_params=sampling_params,
            )
        )

        # find the first text block in the response
        for block in response.content:
            if isinstance(block, TextBlock):
                raw_text = block.text
                # despite the instructions, the LLM might
                # generate markdown like ```sql ... ```
                # so we try to extract the SQL code from it
                # remove the prefix and suffix if present
                return self._extract_sql_code(raw_text, transpile_sql=False)

        raise ValueError("Failed to generate SQL query")

    def _transpile_sql(self, sql: str) -> str:
        if not self._dialect:
            return sql

        for read_dialect in Dialects:
            if read_dialect.value == self._dialect:
                continue

            with contextlib.suppress(ParseError):
                result = "\n".join(
                    sqlglot.transpile(sql, read=read_dialect, write=self._dialect)
                )

                if result:
                    return result

        try:
            return "\n".join(sqlglot.transpile(sql, identity=True, write=self._dialect))
        except ParseError as e:
            error_str = str(e)
            error_str = _ansi_escape.sub("", error_str)
            raise ValueError(f"Generated SQL query is invalid: {error_str}") from e
        except Exception as e:
            raise e

    def _extract_sql_code(self, raw_text: str, transpile_sql: bool = True) -> str:
        """Extract SQL code from the raw text, removing any Markdown formatting.

        LLM usually generates the SQL code wrapped in triple markdown ```
        blocks, sometimes with a "sql" language hint. This function extracts the
        actual SQL code from such formatting.
        """
        # Find the index of the first ``` and last ```
        start_idx = raw_text.find("```")
        end_idx = raw_text.rfind("```")

        clean_code: str

        if start_idx != -1 and end_idx != -1 and start_idx != end_idx:
            # Extract the content between the first and last ```
            code_block = raw_text[start_idx + 3 : end_idx].strip()
            # If the code block starts with "sql", remove it
            if code_block.lower().startswith("sql"):
                code_block = code_block[3:].strip()
            clean_code = code_block.strip()
        else:
            # No code block found, return the original text trimmed
            clean_code = raw_text.strip()

        # remove any ANSI escape sequences, since LLM
        # can sometimes generate them
        result = _ansi_escape.sub("", clean_code)

        # remove final semicolon if present
        if result.endswith(";"):
            result = result[:-1].rstrip()

        # Transpile to the target dialect if needed
        if transpile_sql:
            result = self._transpile_sql(result)

            # remove any ANSI escape sequences, just in case
            # than the transpiler added any
            result = _ansi_escape.sub("", result)

            # remove leading and trailing whitespace
            result = result.strip()

        return result

    def _list_target_schemas(self, include_system: bool) -> list[str]:
        self._ensure_connected()
        meta = inspect(self._engine)
        all_schemas = meta.get_schema_names()  # type: ignore
        if self.schemas:
            selected = [s for s in all_schemas if s in set(self.schemas)]
        else:
            if include_system:
                selected = all_schemas
            else:
                selected = [
                    s for s in all_schemas if s not in _DEFAULT_EXCLUDED_SCHEMAS
                ]

        selected.sort()
        return selected

    def _get_inspectors(self) -> Generator[DatabaseObjectInspector, None, None]:
        if not self.inspector_config or self.inspector_config.views:
            yield DatabaseViewInspector(
                self._engine,
                self._connection,
                self.connection_string,
                self.is_readonly,
            )

        if not self.inspector_config or self.inspector_config.tables:
            yield DatabaseTableInspector(
                self._engine,
                self._connection,
                self.connection_string,
                self.is_readonly,
            )

        if not self.inspector_config or self.inspector_config.procedures:
            yield DatabaseProcedureInspector(
                self._engine,
                self._connection,
                self.connection_string,
                self.is_readonly,
            )

        if not self.inspector_config or self.inspector_config.functions:
            yield DatabaseFunctionsInspector(
                self._engine,
                self._connection,
                self.connection_string,
                self.is_readonly,
            )

    def _get_cached_objects_by_type(
        self,
        cache_key: str,
        schema: str,
        inspector: DatabaseObjectInspector,
    ) -> list[InspectedDatabaseObject]:
        cached_objects = self.cache.get("database-schema", cache_key)
        if cached_objects is not None:
            return cast(list[InspectedDatabaseObject], cached_objects)

        objects = list(inspector.get_objects(schema))
        self.cache.set("database-schema", cache_key, objects)
        return objects

    def _extract_database_schema(
        self,
        include_system: bool = False,
    ) -> list[InspectedSchema]:
        self._ensure_connected()
        result: list[InspectedSchema] = []
        target_schemas = self._list_target_schemas(include_system)

        for schema in target_schemas:
            out_schema: InspectedSchema = InspectedSchema()
            out_schema.name = schema

            for inspector in self._get_inspectors():
                type_cache_key = f"{self.connection_string}_{schema}_{inspector.get_inspector_type()}"

                db_objects = self._get_cached_objects_by_type(
                    type_cache_key, schema, inspector
                )

                for db_object in db_objects:
                    out_schema.add_object(db_object)

            result.append(out_schema)

        cleaned_up_schema = [s for s in result if s.all_objects]
        return cleaned_up_schema

    def _try_to_fix_unsupported_type_error(self, original_sql: str) -> str | None:
        try:
            from sqlgpt_parser.format.formatter import (  # type: ignore[import-not-found,import-untyped]
                format_sql,
            )
            from sqlgpt_parser.parser.mysql_parser import (  # type: ignore[import-not-found,import-untyped]
                parser as mysql_parser,
            )
            from sqlgpt_parser.parser.tree.expression import (  # type: ignore[import-not-found,import-untyped]
                QualifiedNameReference,
            )
            from sqlgpt_parser.parser.tree.qualified_name import (  # type: ignore[import-not-found,import-untyped]
                QualifiedName,
            )
            from sqlgpt_parser.parser.tree.select_item import (  # type: ignore[import-not-found,import-untyped]
                SingleColumn,
            )
        except ImportError:
            logger.warning(
                "sqlgpt_parser is required for fixing unsupported type errors."
            )
            return None

        # 1. Parse SQL to AST
        ast = mysql_parser.parse(original_sql)

        # 2. Extract all table names used in the query (handles both Table and Join)
        def extract_table_names(from_clause: Any) -> list[str]:
            if hasattr(from_clause, "name") and from_clause.name:
                parts: list[str] = list(from_clause.name.parts)
                return parts
            elif hasattr(from_clause, "left") and hasattr(from_clause, "right"):
                result: list[str] = [
                    *extract_table_names(from_clause.left),
                    *extract_table_names(from_clause.right),
                ]
                return result
            return []

        table_parts = extract_table_names(
            ast.query_body.from_
        )  # Return a list with schema and table parts
        table_parts_tolower: list[str] = [p.lower() for p in table_parts]

        # 3. Extract database schema
        extracted_schemas: list[InspectedSchema] = self._extract_database_schema()

        # 4. Filter only tables used in the query
        filtered_tables = [
            item
            for schema in extracted_schemas
            if schema.name.lower() in table_parts_tolower
            for item in schema.tables + schema.views
            if item.name.lower() in table_parts_tolower
        ]
        # 5. Check if query is select *
        select_items = ast.query_body.select.select_items
        is_select_all = (
            len(select_items) == 1
            and isinstance(select_items[0].expression, QualifiedNameReference)
            and select_items[0].expression.name.parts == ["*"]
        )

        if is_select_all:
            # Case SELECT *: put all columns, add ToString to columns of type NULL
            new_select_items = [
                *[
                    SingleColumn(
                        expression=QualifiedNameReference(
                            name=QualifiedName.of(column.name)
                        )
                    )
                    for table in filtered_tables
                    for column in table.columns
                    if column.type
                    != "NULL"  # When mssql returns NULL type, it means the column have unsupported type
                ],
                *[
                    SingleColumn(
                        expression=QualifiedNameReference(
                            name=QualifiedName.of(f"{column.name}.ToString()")
                        )
                    )
                    for table in filtered_tables
                    for column in table.columns
                    if column.type == "NULL"
                ],
            ]
        else:
            # Case specific columns: only modify NULL columns to have .ToString()
            new_select_items = []
            for item in select_items:
                full_col_parts = item.expression.name.parts

                # Extract table prefix and column name
                if len(full_col_parts) > 1:
                    table_prefix = full_col_parts[-2]
                    col_name = full_col_parts[-1]
                else:
                    table_prefix = None
                    col_name = full_col_parts[0]

                # Find the column type among filtered tables
                column_type = None
                for table in filtered_tables:
                    # If there's a table prefix, match it (case-insensitive)
                    if table_prefix and table.name.lower() != table_prefix.lower():
                        continue

                    for column in table.columns:
                        if column.name.lower() == col_name.lower():
                            column_type = column.type
                            break
                    if column_type is not None:
                        break

                # Build the expression preserving the original prefix structure
                col_full_name = ".".join(full_col_parts)
                if (
                    column_type == "NULL"
                ):  # When mssql returns NULL type, it means the column have unsupported type
                    col_full_name += ".ToString()"
                expr_name = QualifiedName.of(col_full_name)

                new_select_items.append(
                    SingleColumn(expression=QualifiedNameReference(name=expr_name))
                )

        # 6. Replace select_items if new ones created
        if len(new_select_items) > 0:
            ast.query_body.select.select_items = new_select_items
            result = format_sql(ast)
            if isinstance(result, str):
                return result
        return None

    # TODO: need a review for postgres cursor support
    def _check_result(self, result: QueryResult) -> QueryResult:
        if result.error is not None:
            return result
        if (
            result.rows
            and result.row_count == 1
            and result.columns
            and len(result.columns) == 1
        ):
            row_value = result.rows[0][0] if result.rows[0] else None

            if isinstance(row_value, str) and result.query and "CALL" in result.query:
                result.error = ErrorQueryResult(
                    type=ErrorType.RETURNED_SQL_CURSOR,
                    description="The query returned a SQL cursor (refcursor) that requires manual fetching. "
                    'Keep your original CALL statement, then use FETCH ALL FROM "<cursor_name>" to retrieve the results. '
                    "Do not use BEGIN, COMMIT, or any other statements—only CALL followed by FETCH.",
                )
                result.rows = []
                result.columns = None
                result.row_count = -1

        return result

    def close(self) -> None:
        """Close any open connections."""
        self._disconnect()

    def _execute_db2_procedure(self, call_statement: str) -> QueryResult:
        try:
            proc_name, param_values, out_indices = self._parse_call_statement(
                call_statement
            )
            conn = self._create_db2_connection()

            try:
                param_metadata = self._get_procedure_param_metadata(conn, proc_name)
                result_params = self._execute_procedure(conn, proc_name, param_values)
                out_values, out_columns = self._extract_output_parameters(
                    result_params, out_indices, param_metadata
                )

                return QueryResult(
                    query=call_statement,
                    rows=[tuple(out_values)] if out_values else [],
                    columns=out_columns,
                    row_count=1 if out_values else 0,
                )

            finally:
                _load_ibm_db().close(conn)

        except Exception as e:
            return QueryResult(
                query=call_statement,
                error=ErrorQueryResult(
                    description=f"DB2 procedure execution failed: {e!s}",
                    type=ErrorType.UNKNOWN,
                ),
                row_count=-1,
            )

    def _parse_call_statement(
        self, call_statement: str
    ) -> tuple[str, list[str | int], list[int]]:
        match = re.search(
            r"CALL\s+([\w.]+)\s*\((.*?)\)", call_statement, re.IGNORECASE | re.DOTALL
        )
        if not match:
            raise ValueError("Invalid CALL statement format")

        proc_name = match.group(1)
        params_str = match.group(2)

        param_values: list[str | int] = []
        out_indices: list[int] = []

        for i, param in enumerate(params_str.split(",")):
            param = param.strip()
            if param == "?":
                param_values.append(0)  # Default value for OUT parameters
                out_indices.append(i)
            else:
                param_values.append(param.strip("'\""))

        return proc_name, param_values, out_indices

    def _create_db2_connection(self) -> Any:
        conn_str = self.connection_string.split("://")[1]
        user_pass, host_db = conn_str.split("@")
        user, password = user_pass.split(":")
        host_port, database = host_db.split("/")
        host, port = host_port.split(":")

        db2_conn_str = (
            f"DATABASE={database};"
            f"HOSTNAME={host};"
            f"PORT={port};"
            f"PROTOCOL=TCPIP;"
            f"UID={user};"
            f"PWD={password};"
        )

        return _load_ibm_db().connect(db2_conn_str, "", "")

    def _execute_procedure(
        self, conn: Any, proc_name: str, param_values: list[str | int]
    ) -> tuple[Any, ...]:
        ibm_db = _load_ibm_db()
        result = ibm_db.callproc(conn, proc_name, tuple(param_values))
        stmt = result[0]
        result_params: tuple[Any, ...] = result[1:]

        # Consume all result sets to ensure OUT parameters are populated
        if stmt:
            while ibm_db.fetch_row(stmt):
                pass
            while ibm_db.next_result(stmt):
                while ibm_db.fetch_row(stmt):
                    pass

        return result_params

    def _get_procedure_param_metadata(
        self, conn: Any, proc_name: str
    ) -> list[tuple[str, str]]:
        schema, name = (
            proc_name.rsplit(".", 1) if "." in proc_name else (None, proc_name)
        )
        ibm_db = _load_ibm_db()
        stmt = ibm_db.procedure_columns(conn, None, schema, name, None)

        params: list[tuple[str, str]] = []
        if stmt:
            row = ibm_db.fetch_assoc(stmt)
            while row:
                param_name = row["COLUMN_NAME"] or f"PARAM_{row['ORDINAL_POSITION']}"
                param_type_code = row["COLUMN_TYPE"]

                type_map = {1: "IN", 2: "INOUT", 4: "OUT"}
                param_type = type_map.get(param_type_code, "UNKNOWN")
                params.append((param_name, param_type))

                row = ibm_db.fetch_assoc(stmt)

        return params

    def _extract_output_parameters(
        self,
        result_params: tuple[Any, ...],
        out_indices: list[int],
        param_metadata: list[tuple[str, str]],
    ) -> tuple[list[Any], list[str]]:
        out_values: list[Any] = [result_params[i] for i in out_indices]
        out_columns: list[str] = []

        for idx in out_indices:
            if idx < len(param_metadata):
                param_name, param_type = param_metadata[idx]
                if param_type in ("OUT", "INOUT"):
                    out_columns.append(param_name)
                else:
                    out_columns.append(f"OUT_PARAM_{idx + 1}")
            else:
                out_columns.append(f"OUT_PARAM_{idx + 1}")

        return out_values, out_columns
