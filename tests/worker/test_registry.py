import pytest

from private_gpt.worker.registry import (
    get_worker_mode,
    register_worker_mode,
    run_worker,
)


def test_registered_application_mode_can_be_dispatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    register_worker_mode("custom", lambda args: calls.extend(args))
    monkeypatch.setenv("PGPT_WORKER_MODE", "CUSTOM")

    run_worker(["custom"])

    assert calls == ["custom"]


def test_worker_mode_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGPT_WORKER_MODE", raising=False)

    with pytest.raises(ValueError, match="PGPT_WORKER_MODE is required"):
        run_worker()


def test_unknown_worker_mode_lists_registered_modes() -> None:
    register_worker_mode("known", lambda args: None)

    with pytest.raises(ValueError, match=r"Registered modes: .*known"):
        get_worker_mode("unknown")
