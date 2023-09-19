"""Tests for `api/main.py`."""
from src.api.main import health_check_route
from src.api.types import HealthRouteOutput


def test_health_check_route() -> None:
    assert health_check_route() == HealthRouteOutput(status="ok")
