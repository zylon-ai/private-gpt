from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from private_gpt.components.code_execution.code_execution_component import (
    CodeExecutionComponent,
)
from private_gpt.components.persistence.persistence_component import (
    PersistenceComponent,
)
from private_gpt.launcher import create_app
from tests.fixtures.fast_api_test_client import inject_global_injector
from tests.fixtures.mock_injector import MockInjector

_SESSION_ID = "test-session-abc123"
_FILE_CONTENT = b"col_a,col_b\n1,2\n3,4\n"
_FILE_NAME = "data.csv"
_MIME_TYPE = "text/plain"  # python-magic detects CSV as text/plain


@pytest.fixture
def volume_root(tmp_path: Path) -> Path:
    """Host-side volume root with pre-created session upload/output dirs."""
    session_path = tmp_path / "sessions" / _SESSION_ID
    (session_path / "uploads").mkdir(parents=True)
    (session_path / "outputs").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def files_client(injector: MockInjector, volume_root: Path) -> TestClient:
    """TestClient configured with a local volume_root and a mocked sandbox."""
    injector.bind_settings({"code_execution": {"volume_root": str(volume_root)}})

    ce_mock = injector.bind_mock(CodeExecutionComponent)
    ce_mock.get_or_create_session = AsyncMock(return_value=None)

    injector.get(PersistenceComponent).apply_migrations()

    app = create_app(injector.test_injector)
    app.middleware("http")(inject_global_injector(injector))
    return TestClient(app)
