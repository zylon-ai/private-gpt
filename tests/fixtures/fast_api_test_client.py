import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from private_gpt.main import app


@pytest.fixture()
def current_test_app() -> FastAPI:
    return app


@pytest.fixture()
def test_client() -> TestClient:
    return TestClient(app)
