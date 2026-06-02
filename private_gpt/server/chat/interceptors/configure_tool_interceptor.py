from injector import inject, singleton

from private_gpt.components.engines.chat_loop.interceptors.chat_loop_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_interceptor_context import (
    ChatLoopInterceptorContext,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_phase import (
    InterceptorPhase,
)
from private_gpt.server.chat.interceptors.null_tool_values_interceptor import (
    NullToolValuesRequestInterceptor,
)
from private_gpt.server.chat.interceptors.schema_coercing_tool_interceptor import (
    SchemaCoercingToolInterceptor,
)


@singleton
class ConfigureToolRequestInterceptor(ChatRequestLoopInterceptor):
    """Aggregate tool-configuration sub-interceptors into a single step."""

    @inject
    def __init__(
        self,
        null_tool_values_interceptor: NullToolValuesRequestInterceptor,
        schema_coercing_interceptor: SchemaCoercingToolInterceptor,
    ) -> None:
        self._interceptors: list[ChatRequestLoopInterceptor] = [
            null_tool_values_interceptor,
            schema_coercing_interceptor,
        ]

    async def intercept(self, context: ChatLoopInterceptorContext) -> None:
        """Run all tool-configuration interceptors in order."""
        if context.phase != InterceptorPhase.BEFORE_ITERATION:
            return

        for interceptor in self._interceptors:
            await interceptor.intercept(context)
