"""Tests for the generic context bag propagation across broker boundaries."""

from __future__ import annotations

from private_gpt.context import current_bag, reinstall, replace_bag, snapshot


def test_current_bag_is_shared_and_isolated_per_context() -> None:
    current_bag()["trace_id"] = "abc"
    assert snapshot() == {"trace_id": "abc"}


def test_snapshot_returns_a_copy() -> None:
    current_bag()["k"] = "v"
    snap = snapshot()
    snap["k"] = "changed"
    assert current_bag()["k"] == "v"


def test_reinstall_installs_transported_bag_and_restores() -> None:
    current_bag()["before"] = "kept"
    transported = {"trace_id": "t1", "headers": {"authorization": "Bearer x"}}
    with reinstall(transported):
        assert current_bag()["trace_id"] == "t1"
        assert current_bag()["headers"]["authorization"] == "Bearer x"
    # Original bag restored after the job.
    assert current_bag()["before"] == "kept"
    assert "trace_id" not in current_bag()


def test_reinstall_none_installs_empty_bag() -> None:
    current_bag()["before"] = "kept"
    with reinstall(None):
        assert current_bag() == {}
    assert current_bag()["before"] == "kept"


def test_reinstall_restores_previous_bag_on_exception() -> None:
    import pytest

    current_bag()["before"] = "kept"
    with pytest.raises(RuntimeError), reinstall({"trace_id": "boom"}):
        raise RuntimeError("boom")
    assert current_bag()["before"] == "kept"


def test_replace_bag_is_copy_on_write_isolating_tasks() -> None:
    """A task created with asyncio.create_task snapshots the ContextVar by
    reference; replace_bag rebinds the var to a new dict so the child's view
    is not mutated by a later parent write."""
    import asyncio

    async def run() -> None:
        current_bag()["who"] = "parent-original"
        seen: dict[str, str] = {}

        async def child() -> None:
            await asyncio.sleep(0)
            seen["who"] = current_bag().get("who", "<missing>")

        task = asyncio.create_task(child())
        # Parent mutates its own bag after the task was created.
        replace_bag({"who": "parent-mutated"})
        await task
        assert seen["who"] == "parent-original"

    asyncio.run(run())
