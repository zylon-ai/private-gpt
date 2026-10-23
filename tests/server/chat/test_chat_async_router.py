from typing import TYPE_CHECKING

import pytest
from httpx import AsyncClient

from private_gpt.chat.input_models import MessageInput
from private_gpt.components.streaming.providers.models import (
    StreamStatus,
)
from private_gpt.server.chat.chat_router import ChatBody
from private_gpt.server.chat_async.chat_async_router import (
    ChatCancellationResponse,
    ChatResponse,
    StreamMetadata,
)

if TYPE_CHECKING:
    from httpx import Response


@pytest.mark.anyio
async def test_initiate_chat_stream(async_test_client: AsyncClient) -> None:
    body = ChatBody(
        messages=[MessageInput(content="Hello", role="user")],
        stream=False,
    )
    response: Response = await async_test_client.post(
        "/v1/messages/async", json=body.model_dump()
    )

    assert response.status_code == 200
    chat_response: ChatResponse = ChatResponse.model_validate(response.json())
    assert chat_response.message_id is not None
    assert str(chat_response.status).lower() == str(StreamStatus.PENDING).lower()
    assert chat_response.message == "Request initiated successfully"


@pytest.mark.anyio
async def test_initiate_chat_stream_with_message_id(
    async_test_client: AsyncClient,
) -> None:
    body = ChatBody(
        messages=[MessageInput(content="Hello", role="user")],
        stream=False,
    )
    message_id: str = "test-correlation-123"
    response: Response = await async_test_client.post(
        f"/v1/messages/async?message_id={message_id}", json=body.model_dump()
    )

    assert response.status_code == 200
    chat_response: ChatResponse = ChatResponse.model_validate(response.json())
    assert chat_response.message_id == message_id


@pytest.mark.anyio
async def test_observe_stream_not_found(async_test_client: AsyncClient) -> None:
    message_id: str = "non-existent-stream"

    response: Response = await async_test_client.get(
        f"/v1/messages/async/{message_id}/stream"
    )

    assert response.status_code == 404
    error_detail: dict = response.json()
    assert "not found" in error_detail["detail"]


@pytest.mark.anyio
async def test_observe_stream_success(async_test_client: AsyncClient) -> None:
    body = ChatBody(
        messages=[MessageInput(content="Hello", role="user")],
        stream=False,
    )

    create_response: Response = await async_test_client.post(
        "/v1/messages/async", json=body.model_dump()
    )
    chat_response: ChatResponse = ChatResponse.model_validate(create_response.json())
    message_id: str = chat_response.message_id

    response: Response = await async_test_client.get(
        f"/v1/messages/async/{message_id}/stream"
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


@pytest.mark.anyio
async def test_cancel_stream_not_found(async_test_client: AsyncClient) -> None:
    message_id: str = "non-existent-stream"

    response: Response = await async_test_client.post(
        f"/v1/messages/async/{message_id}/cancel"
    )

    assert response.status_code == 404
    error_detail: dict = response.json()
    assert "not found" in error_detail["detail"]


@pytest.mark.anyio
async def test_cancel_stream_success(async_test_client: AsyncClient) -> None:
    body = ChatBody(
        messages=[MessageInput(content="Hello", role="user")],
        stream=False,
    )
    create_response: Response = await async_test_client.post(
        "/v1/messages/async", json=body.model_dump()
    )
    chat_response: ChatResponse = ChatResponse.model_validate(create_response.json())
    message_id: str = chat_response.message_id

    response: Response = await async_test_client.post(
        f"/v1/messages/async/{message_id}/cancel"
    )

    assert response.status_code == 200
    cancel_response: ChatCancellationResponse = ChatCancellationResponse.model_validate(
        response.json()
    )
    assert cancel_response.message_id == message_id
    assert cancel_response.message == "Stream cancelled successfully"


@pytest.mark.anyio
async def test_clean_stream_success(async_test_client: AsyncClient) -> None:
    body = ChatBody(
        messages=[MessageInput(content="Hello", role="user")],
        stream=False,
    )
    create_response: Response = await async_test_client.post(
        "/v1/messages/async", json=body.model_dump()
    )
    chat_response: ChatResponse = ChatResponse.model_validate(create_response.json())
    message_id: str = chat_response.message_id

    response: Response = await async_test_client.delete(
        f"/v1/messages/async/{message_id}/delete"
    )

    assert response.status_code == 200


@pytest.mark.anyio
async def test_get_stream_status_not_found(async_test_client: AsyncClient) -> None:
    message_id: str = "non-existent-stream"
    response: Response = await async_test_client.get(
        f"/v1/messages/async/{message_id}/status"
    )

    assert response.status_code == 404
    error_detail: dict = response.json()
    assert "not found" in error_detail["detail"]


@pytest.mark.anyio
async def test_get_stream_status_success(async_test_client: AsyncClient) -> None:
    body = ChatBody(
        messages=[MessageInput(content="Hello", role="user")],
        stream=False,
    )

    create_response: Response = await async_test_client.post(
        "/v1/messages/async", json=body.model_dump()
    )
    chat_response: ChatResponse = ChatResponse.model_validate(create_response.json())
    message_id: str = chat_response.message_id

    response: Response = await async_test_client.get(
        f"/v1/messages/async/{message_id}/status"
    )

    assert response.status_code == 200
    stream_metadata: StreamMetadata = StreamMetadata.model_validate(response.json())
    assert stream_metadata.message_id == message_id


@pytest.mark.anyio
async def test_initiate_chat_stream_validation_error(
    async_test_client: AsyncClient,
) -> None:
    response: Response = await async_test_client.post(
        "/v1/messages/async",
        json={},
    )

    assert response.status_code == 400


@pytest.mark.anyio
async def test_initiate_multiple_chat_streams_unique_message_ids(
    async_test_client: AsyncClient,
) -> None:
    body = ChatBody(
        messages=[MessageInput(content="Hello", role="user")],
        stream=False,
    )

    response1: Response = await async_test_client.post(
        "/v1/messages/async", json=body.model_dump()
    )
    assert response1.status_code == 200
    chat_response1: ChatResponse = ChatResponse.model_validate(response1.json())

    response2: Response = await async_test_client.post(
        "/v1/messages/async", json=body.model_dump()
    )
    assert response2.status_code == 200
    chat_response2: ChatResponse = ChatResponse.model_validate(response2.json())

    assert chat_response1.message_id != chat_response2.message_id


@pytest.mark.anyio
async def test_initiate_chat_stream_duplicate_message_id(
    async_test_client: AsyncClient,
) -> None:
    message_id = "test-correlation-123"
    body = ChatBody(
        messages=[MessageInput(content="Hello", role="user")],
        stream=False,
    )
    response1: Response = await async_test_client.post(
        f"/v1/messages/async?message_id={message_id}", json=body.model_dump()
    )
    assert response1.status_code == 200

    chat_response1: ChatResponse = ChatResponse.model_validate(response1.json())
    assert chat_response1 is not None

    response2: Response = await async_test_client.post(
        f"/v1/messages/async?message_id={message_id}", json=body.model_dump()
    )
    assert response2 is not None
    assert response2.status_code == 400
