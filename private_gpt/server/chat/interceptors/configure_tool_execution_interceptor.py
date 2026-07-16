from injector import inject, singleton

from private_gpt.components.tools.remote_execution import (
    ToolExecutionInterceptor,
    ToolExecutionInterceptorContext,
)
from private_gpt.server.chat.interceptors.null_tool_values_interceptor import (
    NullToolValuesRequestInterceptor,
)
from private_gpt.server.chat.interceptors.schema_coercing_tool_interceptor import (
    SchemaCoercingToolInterceptor,
)


@singleton
class ConfigureToolExecutionInterceptor(ToolExecutionInterceptor):
    """Aggregate tool-execution sub-interceptors into a single step."""

    @inject
    def __init__(
        self,
        null_tool_values_interceptor: NullToolValuesRequestInterceptor,
        schema_coercing_interceptor: SchemaCoercingToolInterceptor,
    ) -> None:
        self._interceptors: list[ToolExecutionInterceptor] = [
            schema_coercing_interceptor,
            null_tool_values_interceptor,
        ]

    async def intercept(self, context: ToolExecutionInterceptorContext) -> None:
        for interceptor in self._interceptors:
            await interceptor.intercept(context)
