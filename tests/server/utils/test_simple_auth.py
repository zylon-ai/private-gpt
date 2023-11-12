"""Tests to validate that the simple authentication mechanism is working.

NOTE: We are not testing the switch based on the config in
      `private_gpt.server.utils.auth`. This is not done because of the way the code
      is currently architecture (it is hard to patch the `settings` and the app while
      the tests are directly importing them).
"""
from typing import Annotated

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient

from private_gpt.server.utils.auth import (
    NOT_AUTHENTICATED,
    _simple_authentication,
    authenticated,
)
from private_gpt.settings.settings import settings


def _copy_simple_authenticated(
    _simple_authentication: Annotated[bool, Depends(_simple_authentication)]
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
    response = test_client.get("/v1/ingest/list")
    assert response.status_code == 401


def test_default_auth_working_when_enabled_200(test_client: TestClient) -> None:
    response_fail = test_client.get("/v1/ingest/list")
    assert response_fail.status_code == 401

    response_success = test_client.get(
        "/v1/ingest/list", headers={"Authorization": settings().server.auth.secret}
    )
    assert response_success.status_code == 200
