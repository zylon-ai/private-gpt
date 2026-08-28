"""Tests for the typed request-scoped tool execution context."""

from __future__ import annotations

from pydantic import BaseModel, Field

from private_gpt.components.sandbox.mount import Mount
from private_gpt.components.tools.tool_execution_context import (
    ToolExecutionContext,
)


class _FakeConfig(BaseModel):
    session_id: str
    mounts: list[Mount] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class _OtherConfig(BaseModel):
    session_id: str
    env: dict[str, str] = Field(default_factory=dict)


def test_from_state_none_when_unavailable() -> None:
    assert ToolExecutionContext.from_state(None) is None


def test_overlay_on_config_with_matching_field() -> None:
    skill = Mount(target="/mnt/skills/x/", access="ro", name="skill-x")
    ctx = ToolExecutionContext(mounts=[skill])
    config = _FakeConfig(session_id="s1")

    updated = ctx.overlay_on(config)

    assert [m.target for m in updated.mounts] == ["/mnt/skills/x/"]
    # Original untouched.
    assert config.mounts == []


def test_overlay_on_config_without_matching_field_is_unchanged() -> None:
    ctx = ToolExecutionContext(mounts=[Mount(target="/mnt/skills/x/", access="ro")])
    config = _OtherConfig(session_id="s1")

    updated = ctx.overlay_on(config)

    assert updated is not config
    assert updated.session_id == "s1"
    assert updated.env == {}


def test_json_roundtrip_preserves_mounts() -> None:
    ctx = ToolExecutionContext(
        mounts=[Mount(target="/mnt/skills/x/", access="ro", name="skill-x")]
    )

    data = ctx.model_dump(mode="json")
    back = ToolExecutionContext.model_validate(data)

    assert [m.target for m in back.mounts] == ["/mnt/skills/x/"]
    assert back.mounts[0].name == "skill-x"
