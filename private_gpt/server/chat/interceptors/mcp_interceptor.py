import asyncio

from httpx import HTTPStatusError
from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import (
    ChatRequest,
    ToolSpec,
)
from private_gpt.components.context.models.context_layer import ToolDefinitionsLayer
from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import (
    InterceptorPhase,
)
from private_gpt.events.event_errors import Errors
from private_gpt.server.mcp.config import McpServerConfig
from private_gpt.server.mcp.mcp_service import McpService


def _extract_original_exception(exc: BaseException) -> BaseException:
    if isinstance(exc, HTTPStatusError):
        if exc.response.status_code in (401, 403):
            return PermissionError(
                f"MCP server rejected the request with HTTP {exc.response.status_code}. "
                "Check your authorization token."
            )
        return exc

    if isinstance(exc, BaseExceptionGroup):
        if len(exc.exceptions) == 1:
            return _extract_original_exception(exc.exceptions[0])
        for sub in exc.exceptions:
            found = _extract_original_exception(sub)
            if found is not exc:
                return found

    return exc


@singleton
class McpRequestInterceptor(ChatRequestLoopInterceptor):
    @inject
    def __init__(self, mcp_service: McpService) -> None:
        self._mcp_service = mcp_service

    async def _collect_tools_from_mcp(
        self,
        request: ChatRequest,
        raise_on_error: bool = True,
    ) -> list[ToolSpec]:
        try:
            output_tools: list[ToolSpec] = []
            if request.mcp_servers:

                async def _get_mcp_tools(
                    config: McpServerConfig,
                ) -> list[ToolSpec]:
                    """Fetch tools from a single MCP server."""
                    client = self._mcp_service.create_client(config)
                    li_tools = await client.list_tools()
                    return [ToolSpec.from_llama_index(tool) for tool in li_tools]

                # Gather tools from all MCP servers concurrently
                mcp_tools_results: list[BaseException | list[ToolSpec]] = list(
                    await asyncio.gather(
                        *(
                            _get_mcp_tools(mcp_server_config)
                            for mcp_server_config in request.mcp_servers
                        ),
                        return_exceptions=True,
                    )
                )

                # Find any issue with the connection
                for result_or_exception in mcp_tools_results:
                    if isinstance(result_or_exception, BaseException):
                        exception = result_or_exception
                        raise _extract_original_exception(exception)

                # flatten the list of tools
                mcp_tools: list[ToolSpec] = [
                    mcp_tool
                    for tools in mcp_tools_results
                    if isinstance(tools, list)
                    for mcp_tool in tools
                ]
                output_tools.extend(mcp_tools)

            return output_tools

        except Exception as e:
            if raise_on_error:
                raise e
            return []

    async def intercept(self, context: ChatInterceptorContext) -> None:
        if (
            context.phase != InterceptorPhase.VALIDATION
            and context.phase != InterceptorPhase.BEFORE_ITERATION
        ):
            return

        try:
            state = context.state
            request = state.input.request

            mcp_tools = await self._collect_tools_from_mcp(request)
            if mcp_tools:
                state.input.context_stack = (
                    state.input.context_stack.remove_layers_of_source("mcp")
                )
                state.input.context_stack = state.input.context_stack.append_layer(
                    ToolDefinitionsLayer(tools=mcp_tools, source="mcp")
                )
                state.input.request.mcp_servers = []

                # Update original context stack as well
                if state.original_input:
                    state.input.context_stack = (
                        state.original_input.context_stack.remove_layers_of_source(
                            "mcp"
                        )
                    )
                    state.original_input.context_stack = (
                        state.original_input.context_stack.remove_layers_of_source(
                            "mcp"
                        ).append_layer(
                            ToolDefinitionsLayer(tools=mcp_tools, source="mcp")
                        )
                    )
                    state.original_input.request.mcp_servers = []

            context.set_state(state)
        except (ImportError, ModuleNotFoundError):
            raise
        except PermissionError as e:
            # Wrap the error to give details about what it is the real error
            raise Errors.build(e, Errors.Codes.PERMISSION_MCP_AUTH_ERROR) from e
        except Exception as e:
            raise Errors.InvalidRequest(
                "Failed to fetch tools from MCP servers.",
                event_code=Errors.Codes.INVALID_REQUEST_INVALID_MCP_ERROR,
            ) from e
