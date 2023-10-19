import pytest
from fastapi.testclient import TestClient

from private_gpt.main import app


@pytest.fixture()
def test_client() -> TestClient:
    return TestClient(app)
