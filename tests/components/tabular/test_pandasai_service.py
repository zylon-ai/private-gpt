import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.components.tabular.pandasai_service import PandasAIService


@pytest.mark.parametrize(
    "error",
    [None, RuntimeError("analysis failed"), asyncio.CancelledError()],
    ids=["success", "failure", "cancellation"],
)
async def test_run_analysis_always_stops_sandbox(error: BaseException | None) -> None:
    sandbox = MagicMock()
    service = MagicMock(spec=PandasAIService)
    service._build_smart_dataframes.return_value = []
    service._resolve_sandbox = AsyncMock(return_value=sandbox)
    result = SimpleNamespace(response=SimpleNamespace(last_code_executed="pass"))
    service._run_analysis_sync.return_value = result
    service._run_analysis_sync.side_effect = error

    if error is None:
        assert await PandasAIService.run_analysis(service, "query") is result
    else:
        with pytest.raises(type(error)):
            await PandasAIService.run_analysis(service, "query")

    sandbox.stop.assert_called_once_with()
