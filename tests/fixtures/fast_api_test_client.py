from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from private_gpt.components.persistence.persistence_component import (
    PersistenceComponent,
)
from private_gpt.di import set_global_injector
from private_gpt.launcher import create_app
from tests.fixtures.mock_injector import MockInjector


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


def _settings_override_from_request(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Resolve settings override passed through indirect fixture parametrization."""
    if hasattr(request, "param") and isinstance(request.param, dict):
        return request.param

    callspec = getattr(request.node, "callspec", None)
    if callspec is None:
        return {}

    # Indirect parametrization in tests is usually attached to `test_client` /
    # `async_test_client` (not `applied_migrations`), so we read from callspec
    # to propagate those overrides before app creation and migrations.
    for fixture_name in ("applied_migrations", "test_client", "async_test_client"):
        value = callspec.params.get(fixture_name)
        if isinstance(value, dict):
            return value
    return {}


def middleware_injector(
    request: pytest.FixtureRequest, injector: MockInjector
) -> MockInjector:
    """Fixture to inject the middleware into the test client."""
    set_global_injector(injector.test_injector)


def inject_global_injector(injector: MockInjector) -> None:
    """Inject the global injector for the test session."""

    async def inject_injector_middleware(request: Request, call_next: Any) -> Any:
        nonlocal injector
        """Middleware to inject the injector into the request state."""
        set_global_injector(injector.test_injector)
        response = await call_next(request)
        return response

    return inject_injector_middleware


@pytest.fixture
def applied_migrations(
    request: pytest.FixtureRequest, injector: MockInjector
) -> MockInjector:
    injector.bind_settings(_settings_override_from_request(request))
    injector.get(PersistenceComponent).apply_migrations()
    return injector


@pytest.fixture
def test_client(
    request: pytest.FixtureRequest, applied_migrations: MockInjector
) -> TestClient:
    injector = applied_migrations

    app_under_test = create_app(injector.test_injector)
    app_under_test.middleware("http")(inject_global_injector(injector))

    return TestClient(app_under_test)


@pytest.fixture
async def async_test_client(
    request: pytest.FixtureRequest, applied_migrations: MockInjector
) -> AsyncClient:
    injector = applied_migrations
    app_under_test = create_app(injector.test_injector)
    app_under_test.middleware("http")(inject_global_injector(injector))

    async with AsyncClient(
        transport=ASGITransport(app=app_under_test),
        base_url="http://test",
    ) as client:
        yield client
