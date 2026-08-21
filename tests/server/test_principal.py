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
        assert Principal.current().authorization_value == "sk-abc"
        assert Principal.current().authorization_prefix == "Bearer"
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
        assert Principal.current().authorization_value == "sk-job"
    # Restored after the job.
    assert Principal.current().anonymous is True


def test_principal_survives_broker_hop_via_bag_with_api_key() -> None:
    Principal(headers={"x-api-key": "sk-header"}).set_current()
    try:
        bag = snapshot()
    finally:
        Principal.reset()

    with reinstall(bag):
        assert Principal.current().api_key == "sk-header"
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


def test_authorization_prefix_and_value() -> None:
    assert (
        Principal(headers={"authorization": "Bearer sk-abc"}).authorization_prefix
        == "Bearer"
    )
    assert (
        Principal(headers={"authorization": "Bearer sk-abc"}).authorization_value
        == "sk-abc"
    )
    assert (
        Principal(headers={"authorization": "Basic dXNlcjpwYXNz"}).authorization_prefix
        == "Basic"
    )
    assert (
        Principal(headers={"authorization": "Basic dXNlcjpwYXNz"}).authorization_value
        == "dXNlcjpwYXNz"
    )
    assert (
        Principal(headers={"authorization": "Digest qop=auth"}).authorization_value
        == "qop=auth"
    )
    # No prefix/value shape — value is the raw header; prefix is None.
    assert Principal(headers={"authorization": "sk-abc"}).authorization_prefix is None
    assert Principal(headers={"authorization": "sk-abc"}).authorization_value == (
        "sk-abc"
    )
    # Empty value after the prefix yields empty (not the raw header).
    assert Principal(headers={"authorization": "Bearer "}).authorization_value == ""
    assert Principal().authorization_prefix is None
    assert Principal().authorization_value is None


def test_api_key_is_distinct_from_authorization() -> None:
    """``api_key`` is the ``x-api-key`` value, independent of the bearer."""
    p = Principal(
        headers={
            "authorization": "Bearer sk-bearer",
            "x-api-key": "sk-header",
        }
    )
    assert p.api_key == "sk-header"
    assert p.authorization_value == "sk-bearer"
    assert p.authorization_prefix == "Bearer"
    # No x-api-key header → api_key is None (not the bearer value).
    assert Principal(headers={"authorization": "Bearer sk-abc"}).api_key is None
    assert Principal().api_key is None


def test_as_env_emits_bare_auth_token() -> None:
    """The sandbox receives the bare value; the prefix stays rebuildable."""
    p = Principal(headers={"authorization": "Bearer sk-abc"})
    assert p.as_env() == {"ANTHROPIC_AUTH_TOKEN": "sk-abc"}
    # Prefix + value rebuild the full Authorization line.
    assert f"{p.authorization_prefix} {p.as_env()['ANTHROPIC_AUTH_TOKEN']}" == (
        "Bearer sk-abc"
    )


def test_as_env_handles_basic_and_api_key() -> None:
    p = Principal(
        headers={
            "authorization": "Basic dXNlcjpwYXNz",
            "x-api-key": "sk-header",
        }
    )
    env = p.as_env()
    assert env == {
        "ANTHROPIC_API_KEY": "sk-header",
        "ANTHROPIC_AUTH_TOKEN": "dXNlcjpwYXNz",
    }
    assert f"{p.authorization_prefix} {env['ANTHROPIC_AUTH_TOKEN']}" == (
        "Basic dXNlcjpwYXNz"
    )
