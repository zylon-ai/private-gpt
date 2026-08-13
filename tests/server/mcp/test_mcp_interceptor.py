from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.components.chat.models.chat_config_models import ChatRequest
from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.events.models import McpTokensRefreshedEvent
from private_gpt.server.chat.interceptors.mcp_interceptor import (
    McpRequestInterceptor,
)
from private_gpt.server.mcp.config import McpServerConfig
from private_gpt.server.mcp.mcp_service import McpToolDefinition


@pytest.mark.asyncio
async def test_discovery_passes_chat_event_callback_to_mcp_client() -> None:
    config = McpServerConfig(
        url="https://mcp.example.com",
        artifact_id="artifact-123",
    )
    client = MagicMock()
    client.list_tools = AsyncMock(
        return_value=[
            McpToolDefinition(
                name="lookup",
                description="Look something up",
                input_schema={"type": "object", "properties": {}},
            )
        ]
    )
    client.close = AsyncMock()
    mcp_service = MagicMock()
    mcp_service.create_client.return_value = client
    interceptor = McpRequestInterceptor(mcp_service)
    request = ChatRequest(messages=[], mcp_servers=[config])
    context = MagicMock()
    context.phase = InterceptorPhase.VALIDATION
    context.state = SimpleNamespace(
        input=SimpleNamespace(request=request, context_stack=MagicMock()),
        original_input=None,
    )
    context.emit_event = MagicMock()

    await interceptor.intercept(context)

    mcp_service.create_client.assert_called_once_with(
        config,
        emit_event=context.emit_event,
    )
    callback = mcp_service.create_client.call_args.kwargs["emit_event"]
    callback(
        McpTokensRefreshedEvent(
            artifact_id="artifact-123",
            previous_refresh_token="refresh-before",
            authorization_token="access-after",
            refresh_token="refresh-after",
        )
    )
    context.emit_event.assert_called_once()
    client.close.assert_awaited_once()
