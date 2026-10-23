import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.server.utils.http_disconnect import cancel_on_http_disconnect


@pytest.mark.anyio
async def test_http_disconnect_cancels_operation() -> None:
    request = MagicMock()
    request.is_disconnected = AsyncMock(return_value=True)
    cancelled = asyncio.Event()

    async def operation() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    with pytest.raises(asyncio.CancelledError):
        await cancel_on_http_disconnect(request, operation(), poll_interval=0)

    assert cancelled.is_set()


@pytest.mark.anyio
async def test_completed_operation_wins_disconnect_race() -> None:
    request = MagicMock()
    request.is_disconnected = AsyncMock(return_value=True)

    async def operation() -> str:
        return "completed"

    result = await cancel_on_http_disconnect(request, operation(), poll_interval=0)

    assert result == "completed"
