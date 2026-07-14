import asyncio
from typing import TYPE_CHECKING, Any, Literal, cast

from injector import inject, singleton
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)
from llama_index.core.base.llms.types import (
    TextBlock as LITextBlock,
)

from private_gpt.chat.input_models import BlobVisibilityMode
from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.tools.binary_block_decorators import (
    auto_resolve_media_blocks,
)
from private_gpt.components.tools.remote_execution import build_rebuild_metadata
from private_gpt.components.tools.tool_names import DATABASE_QUERY_TOOL_NAME
from private_gpt.components.tools.tool_placeholders import DATABASE_QUERY_TOOL_FN
from private_gpt.components.tools.types import ToolValidationMode
from private_gpt.di import get_global_injector
from private_gpt.events.models import (
    BinaryBlock,
    ResultContentBlockType,
    TextBlock,
)
from private_gpt.server.utils.artifact_input import SqlDatabaseArtifact
from private_gpt.settings.settings import Settings
from private_gpt.utils.dependencies import format_missing_dependency_message

if TYPE_CHECKING:
    from private_gpt.components.tabular.database_query_generator import (
        DatabaseQueryGenerator,
        ErrorQueryResult,
        QueryResult,
    )


def _load_database_query_dependencies() -> tuple[
    type["DatabaseQueryGenerator"],
    type["ErrorQueryResult"],
    type["QueryResult"],
]:
    try:
        from private_gpt.components.tabular.database_query_generator import (
            DatabaseQueryGenerator,
            ErrorQueryResult,
            QueryResult,
        )
    except ImportError as e:
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

    return DatabaseQueryGenerator, ErrorQueryResult, QueryResult


@singleton
class DatabaseQueryToolBuilder:
    """A builder class for creating a database query tool.

    This tool allows users to run natural
    language queries against connected SQL databases.
    It runs the same query against all
    provided databases and combines the results.
    """

    @inject
    def __init__(
        self,
        settings: Settings,
        llm_component: LLMComponent,
    ):
        """Initialize the DatabaseQueryToolBuilder with necessary components."""
        self.settings = settings
        self.llm_component = llm_component
        self.sample_size = (
            # number of characters to sample from the result for display
            # TODO: this should be moved to tokens instead
            500
        )

    @staticmethod
    def _get_additional_context(
        chat_history: list[ChatMessage] | None = None,
    ) -> str | None:
        if not chat_history:
            return None

        additional_context: str = ""

        # Add system prompt if available
        system_prompt = next(
            (msg for msg in chat_history if msg.role == MessageRole.SYSTEM),
            None,
        )
        if system_prompt and system_prompt.content:
            additional_context += (
                f"**System Instructions**\n{system_prompt.content}\n\n"
            )

        # Add last user message if available
        last_user_message = next(
            (msg for msg in reversed(chat_history) if msg.role == MessageRole.USER),
            None,
        )
        if last_user_message and last_user_message.content:
            additional_context += (
                f"**Current User Request\n{last_user_message.content}**\n\n"
            )

        # Add last database results if available
        before_user_message = next(
            (
                msg
                for msg in reversed(chat_history)
                if msg.role == MessageRole.USER and (msg != last_user_message)
            ),
            None,
        )
        last_iteration_messages = chat_history.copy() or []
        if last_iteration_messages:
            if last_user_message:
                index = (
                    last_iteration_messages.index(last_user_message)
                    if last_user_message in last_iteration_messages
                    else -1
                )
                if index != -1:
                    last_iteration_messages = last_iteration_messages[:index]

            if before_user_message:
                index = (
                    last_iteration_messages.index(before_user_message)
                    if before_user_message in last_iteration_messages
                    else -1
                )
                if index != -1:
                    last_iteration_messages = last_iteration_messages[index:]

        database_result_messages = [
            msg
            for msg in last_iteration_messages
            if msg.role == MessageRole.TOOL
            and msg.additional_kwargs.get("tool_call_name") == DATABASE_QUERY_TOOL_NAME
        ]

        if database_result_messages:
            # Take only the 1st block
            # 1. Contains the query or an error
            # 2. If apply, it contains the result summary
            results = [
                block.text
                for message in database_result_messages
                for block in message.blocks
                if isinstance(block, LITextBlock)
            ]
            if results:
                additional_context += (
                    "** Previous Database Query (maybe it's relevant):**\n"
                )
                results_str = "\n".join(results)
                additional_context += f"\n```\n{results_str}\n```\n\n"

        additional_context = additional_context.strip()
        return additional_context or None

    async def build_tool(
        self,
        sql_artifacts: list[SqlDatabaseArtifact],
        chat_history: list[ChatMessage] | None = None,
        name: str = DATABASE_QUERY_TOOL_NAME,
        type: str = DATABASE_QUERY_TOOL_NAME + "_v1",
        description: str = DATABASE_QUERY_TOOL_FN.metadata.description,
        validate: ToolValidationMode = ToolValidationMode.LAZY,
        runtime: Literal["client", "server"] = "server",
        blob_visibility: BlobVisibilityMode = BlobVisibilityMode.PUBLIC,
    ) -> ToolSpec:
        (
            database_query_generator_cls,
            error_query_result_cls,
            query_result_cls,
        ) = _load_database_query_dependencies()

        sample_size = self.sample_size  # capture for closure

        async def validate_sql() -> None:
            if not sql_artifacts:
                raise ValueError("At least one SQL database artifact is required.")

            query_gen = [
                database_query_generator_cls(
                    connection_string=sql_artifact.connection_string,
                    ssl=sql_artifact.ssl,
                    schemas=sql_artifact.schemas,
                    # TODO: Meanwhile we validate the feature,
                    # we set readonly to true to avoid accidental data changes
                    is_readonly=True,
                    enable_tables=sql_artifact.enable_tables,
                    enable_views=sql_artifact.enable_views,
                    enable_functions=sql_artifact.enable_functions,
                    enable_procedures=sql_artifact.enable_procedures,
                    description=sql_artifact.description,
                    batch_size=self.settings.database_query.batch_size,
                    timeout_seconds=self.settings.database_query.timeout_seconds,
                    max_mb_result=self.settings.database_query.max_mb_result,
                )
                for sql_artifact in sql_artifacts
            ]
            validations = [gen.check_connection() for gen in query_gen]
            results = list(await asyncio.gather(*validations, return_exceptions=True))
            errors = [
                f"- DB {i} connection error: {result!s}\n"
                for i, result in enumerate(results)
                if isinstance(result, str | Exception)
            ]
            if errors:
                raise ConnectionError(
                    "One or more database connections failed: \n" + "; ".join(errors),
                )

        @auto_resolve_media_blocks(blob_visibility=blob_visibility)
        async def execute_sql(query: str) -> list[ResultContentBlockType]:
            additional_context: str | None = await asyncio.to_thread(
                self._get_additional_context,
                chat_history,
            )
            query_gen = []
            try:
                query_gen = [
                    database_query_generator_cls(
                        connection_string=sql_artifact.connection_string,
                        ssl=sql_artifact.ssl,
                        schemas=sql_artifact.schemas,
                        enable_tables=sql_artifact.enable_tables,
                        enable_views=sql_artifact.enable_views,
                        enable_functions=sql_artifact.enable_functions,
                        enable_procedures=sql_artifact.enable_procedures,
                        description=sql_artifact.description,
                        batch_size=self.settings.database_query.batch_size,
                        timeout_seconds=self.settings.database_query.timeout_seconds,
                        max_mb_result=self.settings.database_query.max_mb_result,
                    )
                    for sql_artifact in sql_artifacts
                ]

                # create the query coroutines
                searches = [
                    gen.query(
                        query=query,
                        additional_context=additional_context,
                    )
                    for gen in query_gen
                ]
                query_results_or_exceptions = await asyncio.gather(
                    *searches, return_exceptions=True
                )
                query_results = []
                for result in query_results_or_exceptions:
                    if isinstance(result, BaseException):
                        query_results.append(
                            query_result_cls(
                                query=None,
                                row_count=-1,
                                error=error_query_result_cls(str(result)),
                            )
                        )
                    else:
                        query_results.append(result)
                if not query_results:
                    return [TextBlock(text="No databases returned results.")]

                # if all row counts are 0, return no results found
                all_failed = all(result.row_count < 0 for result in query_results)
                if all_failed:
                    return [
                        TextBlock(text=f"No results found.\nError: {result.error}")
                        for result in query_results
                    ]

                results = list(zip(sql_artifacts, query_results, strict=False))
                query_with_results = [
                    (sql_artifact, db_query_result)
                    for sql_artifact, db_query_result in results
                    if db_query_result.row_count > 0
                ]
                if query_with_results:
                    # As the model cannot understand that some
                    # databases returned no results,
                    # we only show the databases that returned results
                    # if at least one database returned results
                    results = query_with_results

                result_as_block_list: list[list[ResultContentBlockType]] = []
                for sql_artifact, db_query_result in results:
                    prefix = (
                        f"Database: {sql_artifact.connection_string}\n"
                        if len(results) > 1
                        else ""
                    )
                    blocks: list[ResultContentBlockType] = [
                        TextBlock(
                            text=prefix
                            + "Query:\n```sql\n"
                            + (db_query_result.query or "No query was generated.")
                            + "\n```\n\n Row Count: "
                            + str(db_query_result.row_count)
                        )
                    ]

                    if db_query_result.warning:
                        blocks.append(
                            TextBlock(text="Warning: \n" + str(db_query_result.warning))
                        )

                    if db_query_result.row_count > 0:
                        csv = db_query_result.as_csv()
                        filename = f"csv_{hash(query)}.csv"
                        csv_block = BinaryBlock.from_text(
                            text=csv,
                            filename=filename,
                            mime_type="text/csv",
                        )
                        blocks.append(csv_block)

                        if len(csv) > sample_size:
                            blocks.append(
                                TextBlock(
                                    text="Representative data from the query result. Information IS NOT COMPLETE, refer"
                                    "to the generated csv file for the full response:\n"
                                    + (csv[0:sample_size])
                                )
                            )
                        else:
                            blocks.append(TextBlock(text="Query result: \n" + csv))
                    elif db_query_result.query and db_query_result.row_count == 0:
                        blocks.append(
                            TextBlock(
                                text="Query executed successfully. No rows found."
                            )
                        )
                    elif db_query_result.error:
                        blocks.append(
                            TextBlock(text="Error: \n" + str(db_query_result.error))
                        )

                    result_as_block_list.append(blocks)

                flattened_blocks: list[ResultContentBlockType] = [
                    block for sublist in result_as_block_list for block in sublist
                ]
                if not flattened_blocks:
                    flattened_blocks = [
                        TextBlock(text="No results found for the query.")
                    ]

                return flattened_blocks
            finally:
                # close connections
                def close() -> None:
                    for gen in query_gen:
                        gen.close()

                await asyncio.to_thread(close)

        async def run_tool(query: str) -> list[ResultContentBlockType]:
            if validate == ToolValidationMode.LAZY:
                await validate_sql()

            return await execute_sql(query)

        if validate == ToolValidationMode.EAGER:
            await validate_sql()

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime=runtime,
            description=description,
            async_fn=run_tool,
            execution_metadata=build_rebuild_metadata(
                rebuild_database_query_tool,
                {
                    "sql_artifacts": sql_artifacts,
                    "chat_history": chat_history,
                    "name": name,
                    "type": type,
                    "description": description,
                    "validate": validate,
                    "runtime": runtime,
                    "blob_visibility": blob_visibility,
                },
            ),
        )


async def rebuild_database_query_tool(**kwargs: Any) -> ToolSpec:
    builder = get_global_injector().get(DatabaseQueryToolBuilder)
    return await builder.build_tool(**cast(Any, kwargs))
