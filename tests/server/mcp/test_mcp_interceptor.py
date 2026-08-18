from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest

from private_gpt.components.chat.models.chat_config_models import ChatRequest
from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.events.event_errors import Errors
from private_gpt.events.models import (
    McpTokensRefreshedEvent,
    McpTokensRefreshFailedEvent,
)
from private_gpt.server.chat.interceptors.mcp_interceptor import (
    McpRequestInterceptor,
)
from private_gpt.server.mcp.config import McpServerConfig
from private_gpt.server.mcp.mcp_service import McpToolDefinition


def _discovery_setup() -> tuple[
    McpServerConfig,
    MagicMock,
    MagicMock,
    McpRequestInterceptor,
    MagicMock,
]:
    config = McpServerConfig(
        url="https://mcp.example.com",
        refresh_token="refresh-before",
        client_id="client-id",
        metadata={"artifact_id": "artifact-123"},
    )
    client = MagicMock()
    client.close = AsyncMock()
    client.token_refresh_event.return_value = None
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
    return config, client, mcp_service, interceptor, context


@pytest.mark.asyncio
async def test_discovery_emits_refreshed_tokens_as_a_chat_event() -> None:
    config, client, mcp_service, interceptor, context = _discovery_setup()
    client.list_tools = AsyncMock(
        return_value=[
            McpToolDefinition(
                name="lookup",
                description="Look something up",
                input_schema={"type": "object", "properties": {}},
            )
        ]
    )
    client.token_refresh_event.return_value = McpTokensRefreshedEvent(
        name="mcp",
        url="https://mcp.example.com",
        previous_refresh_token="refresh-before",
        authorization_token="access-after",
        refresh_token="refresh-after",
        metadata={"artifact_id": "artifact-123"},
    )

    await interceptor.intercept(context)

    mcp_service.create_client.assert_called_once_with(config)
    context.emit_event.assert_called_once_with(
        McpTokensRefreshedEvent(
            name="mcp",
            url="https://mcp.example.com",
            previous_refresh_token="refresh-before",
            authorization_token="access-after",
            refresh_token="refresh-after",
            metadata={"artifact_id": "artifact-123"},
        )
    )
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_discovery_emits_refreshed_tokens_before_wrapping_error() -> None:
    _, client, _, interceptor, context = _discovery_setup()
    error = RuntimeError("discovery failed after refresh")
    client.list_tools = AsyncMock(side_effect=error)
    client.token_refresh_event.return_value = McpTokensRefreshedEvent(
        name="mcp",
        url="https://mcp.example.com",
        previous_refresh_token="refresh-before",
        authorization_token="access-after",
        refresh_token="refresh-after",
        metadata={"artifact_id": "artifact-123"},
    )

    with pytest.raises(Errors.InvalidRequest) as exc_info:
        await interceptor.intercept(context)

    assert exc_info.value.__cause__ is error
    context.emit_event.assert_called_once_with(
        McpTokensRefreshedEvent(
            name="mcp",
            url="https://mcp.example.com",
            previous_refresh_token="refresh-before",
            authorization_token="access-after",
            refresh_token="refresh-after",
            metadata={"artifact_id": "artifact-123"},
        )
    )
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_discovery_emits_refresh_failed_before_wrapping_error() -> None:
    _, client, _, interceptor, context = _discovery_setup()
    error = RuntimeError("refresh failed")
    client.list_tools = AsyncMock(side_effect=error)
    client.token_refresh_event.return_value = McpTokensRefreshFailedEvent(
        name="mcp",
        url="https://mcp.example.com",
        error="MCP OAuth token refresh failed",
        metadata={"artifact_id": "artifact-123"},
    )

    with pytest.raises(Errors.InvalidRequest) as exc_info:
        await interceptor.intercept(context)

    assert exc_info.value.__cause__ is error
    context.emit_event.assert_called_once_with(
        McpTokensRefreshFailedEvent(
            name="mcp",
            url="https://mcp.example.com",
            error="MCP OAuth token refresh failed",
            metadata={"artifact_id": "artifact-123"},
        )
    )
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_discovery_without_refresh_keeps_existing_error_path() -> None:
    _, client, _, interceptor, context = _discovery_setup()
    error = RuntimeError("ordinary MCP failure")
    client.list_tools = AsyncMock(side_effect=error)

    with pytest.raises(Errors.InvalidRequest) as exc_info:
        await interceptor.intercept(context)

    assert exc_info.value.status_code == 400
    assert exc_info.value.event_code == Errors.Codes.INVALID_REQUEST_INVALID_MCP_ERROR
    assert exc_info.value.__cause__ is error
    context.emit_event.assert_not_called()
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_discovery_keeps_403_permission_error_path() -> None:
    _, client, _, interceptor, context = _discovery_setup()
    request = httpx2.Request("POST", "https://mcp.example.com")
    response = httpx2.Response(403, request=request)
    client.list_tools = AsyncMock(
        side_effect=httpx2.HTTPStatusError(
            "Forbidden",
            request=request,
            response=response,
        )
    )

    with pytest.raises(Errors.PermissionDenied) as exc_info:
        await interceptor.intercept(context)

    assert exc_info.value.status_code == 403
    assert exc_info.value.event_code == Errors.Codes.PERMISSION_MCP_AUTH_ERROR
    assert isinstance(exc_info.value.original_exception, PermissionError)
    assert "HTTP 403" in str(exc_info.value)
    context.emit_event.assert_not_called()
    client.close.assert_awaited_once()
