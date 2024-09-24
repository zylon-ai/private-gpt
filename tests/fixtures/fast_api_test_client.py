import pytest
from fastapi.testclient import TestClient

from private_gpt.launcher import create_app
from tests.fixtures.mock_injector import MockInjector


@pytest.fixture
def test_client(request: pytest.FixtureRequest, injector: MockInjector) -> TestClient:
    if request is not None and hasattr(request, "param"):
        injector.bind_settings(request.param or {})

    app_under_test = create_app(injector.test_injector)
    return TestClient(app_under_test)
