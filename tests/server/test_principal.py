"""Tests for Principal and the shared context bag.

The principal is now backed by the generic context bag (``private_gpt.context``)
that travels with ARQ/Celery jobs, so these tests verify both the typed claims
and the bag round-trip across a broker hop.
"""

from __future__ import annotations

from private_gpt.context import reinstall, snapshot
from private_gpt.server.principal import Principal


def test_principal_reads_and_writes_the_context_bag() -> None:
    p = Principal(headers={"authorization": "Bearer sk-abc"})
    p.set_current()
    try:
        assert Principal.current().authorization == "Bearer sk-abc"
        assert Principal.current().api_key == "sk-abc"
        # The underlying bag carries the data.
        assert snapshot()["headers"]["authorization"] == "Bearer sk-abc"
    finally:
        Principal.reset()


def test_principal_reset_clears_claims_from_bag() -> None:
    Principal(headers={"authorization": "Bearer sk-abc"}).set_current()
    try:
        assert Principal.current().anonymous is False
    finally:
        Principal.reset()
    assert Principal.current().anonymous is True
    assert "headers" not in snapshot()


def test_anonymous_principal_produces_empty_snapshot() -> None:
    Principal.reset()
    assert Principal.current().anonymous is True
    assert snapshot() == {}


def test_principal_survives_broker_hop_via_bag() -> None:
    """Simulate an ARQ/Celery dispatch: snapshot on the API side, reinstall on
    the worker, and read the principal there."""
    Principal(headers={"authorization": "Bearer sk-job"}).set_current()
    try:
        bag = snapshot()
    finally:
        Principal.reset()

    # Worker side — the transported bag is reinstalled.
    with reinstall(bag):
        assert Principal.current().authorization == "Bearer sk-job"
        assert Principal.current().api_key == "sk-job"
    # Restored after the job.
    assert Principal.current().anonymous is True


def test_principal_survives_broker_hop_via_bag_with_api_key() -> None:
    Principal(headers={"x-api-key": "sk-header"}).set_current()
    try:
        bag = snapshot()
    finally:
        Principal.reset()

    with reinstall(bag):
        assert Principal.current().api_key_header == "sk-header"
    assert Principal.current().anonymous is True


def test_resolve_env_with_headers_on_worker() -> None:
    """The sentinels are resolved after a broker hop because the headers are in
    the bag — this is the production code-execution scenario."""
    Principal(headers={"authorization": "Bearer sk-abc"}).set_current()
    try:
        bag = snapshot()
    finally:
        Principal.reset()

    env = {
        "ANTHROPIC_BASE_URL": "http://backend/gpt",
        "ANTHROPIC_API_KEY": "$PRINCIPAL_API_KEY",
        "ANTHROPIC_AUTH_TOKEN": "$PRINCIPAL_BEARER",
    }
    with reinstall(bag):
        resolved = Principal.current().resolve_env(env)
    assert resolved == {
        "ANTHROPIC_BASE_URL": "http://backend/gpt",
        "ANTHROPIC_AUTH_TOKEN": "sk-abc",
    }
