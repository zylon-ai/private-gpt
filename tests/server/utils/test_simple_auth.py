"""Tests to validate that the simple authentication mechanism is working.

NOTE: We are not testing the switch based on the config in
      `private_gpt.server.utils.auth`. This is not done because of the way the code
      is currently architecture (it is hard to patch the `settings` and the app while
      the tests are directly importing them).
"""

from typing import Annotated
from unittest.mock import AsyncMock

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient

from private_gpt.components.web.web_search.models import WebSearchResult
from private_gpt.components.web.web_search.providers.parallel import (
    ParallelSearchProvider,
)
from private_gpt.server.utils.auth import (
    NOT_AUTHENTICATED,
    _simple_authentication,
    authenticated,
)
from private_gpt.settings.settings import settings


def _copy_simple_authenticated(
    _simple_authentication: Annotated[bool, Depends(_simple_authentication)],
) -> bool:
    """Check if the request is authenticated."""
    if not _simple_authentication:
        raise NOT_AUTHENTICATED
    return True


@pytest.fixture(autouse=True)
def _patch_authenticated_dependency(test_client: TestClient):
    # Patch the server to use simple authentication

    test_client.app.dependency_overrides[authenticated] = _copy_simple_authenticated

    # Call the actual test
    yield

    # Remove the patch for other tests
    test_client.app.dependency_overrides = {}


def test_default_auth_working_when_enabled_401(test_client: TestClient) -> None:
    response = test_client.get("/v1/artifacts/list?collection=default")
    assert response.status_code == 401


def test_default_auth_working_when_enabled_200(test_client: TestClient) -> None:
    response_fail = test_client.get("/v1/artifacts/list?collection=default")
    assert response_fail.status_code == 401

    response_success = test_client.get(
        "/v1/artifacts/list?collection=default",
        headers={"Authorization": settings().server.auth.secret},
    )
    assert response_success.status_code == 200


@pytest.mark.parametrize(
    "test_client", [{"web_search": {"provider": "auto"}}], indirect=True
)
def test_direct_web_search_requires_api_auth_but_no_chat_approval(
    test_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    search = AsyncMock(
        return_value=[
            WebSearchResult(
                url="https://example.org/voyager",
                title="Voyager",
                description="Voyager excerpt",
            )
        ]
    )
    monkeypatch.setattr(ParallelSearchProvider, "make_query", search)

    response = test_client.post(
        "/v1/tools/web-search", json={"query": "Voyager mission"}
    )
    assert response.status_code == 401
    search.assert_not_awaited()

    response = test_client.post(
        "/v1/tools/web-search",
        headers={"Authorization": settings().server.auth.secret},
        json={"query": "Voyager mission"},
    )
    assert response.status_code == 200
    assert response.json()["is_error"] is False
    assert "Voyager excerpt" in response.text
    search.assert_awaited_once()
    assert search.await_args.args[0] == "Voyager mission"
