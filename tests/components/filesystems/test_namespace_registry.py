"""Tests for the filesystem namespace registry (T1.1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from private_gpt.components.filesystems.namespace_registry import NamespaceRegistry
from private_gpt.settings.settings import FilesystemsSettings, NamespaceConfig


def _make_registry(tmp_path: Path, namespaces: dict) -> NamespaceRegistry:
    """Build a NamespaceRegistry from a settings dict, creating roots as needed."""
    ns_configs: dict[str, NamespaceConfig] = {}
    for name, cfg in namespaces.items():
        root = cfg.get("root", "")
        if root and root != "__nonexistent__":
            ns_dir = tmp_path / root
            ns_dir.mkdir(parents=True, exist_ok=True)
            cfg = {**cfg, "root": str(ns_dir)}
        ns_configs[name] = NamespaceConfig(**cfg)

    fs_settings = FilesystemsSettings(namespaces=ns_configs)

    class _FakeSettings:
        filesystems = fs_settings

    return NamespaceRegistry(settings=_FakeSettings())  # type: ignore[arg-type]


class TestRegistryLoad:
    def test_loads_valid_namespace(self, tmp_path: Path) -> None:
        reg = _make_registry(
            tmp_path, {"session": {"root": "vols", "default_mode": "rw"}}
        )
        assert "session" in reg.all_names()

    def test_root_returns_path(self, tmp_path: Path) -> None:
        reg = _make_registry(
            tmp_path, {"session": {"root": "vols", "default_mode": "rw"}}
        )
        assert reg.root("session").exists()

    def test_multiple_namespaces(self, tmp_path: Path) -> None:
        reg = _make_registry(
            tmp_path,
            {
                "session": {"root": "vols/session", "default_mode": "rw"},
                "artifacts": {"root": "vols/artifacts", "default_mode": "rw"},
                "skills": {"root": "vols/skills", "default_mode": "ro"},
            },
        )
        assert sorted(reg.all_names()) == ["artifacts", "session", "skills"]
        assert reg.get("skills").default_mode == "ro"


class TestUnknownNamespace:
    def test_unknown_raises_key_error(self, tmp_path: Path) -> None:
        reg = _make_registry(
            tmp_path, {"session": {"root": "vols", "default_mode": "rw"}}
        )
        with pytest.raises(KeyError, match="unknown"):
            reg.get("unknown")

    def test_empty_all_names_when_no_namespaces(self, tmp_path: Path) -> None:
        reg = _make_registry(tmp_path, {})
        assert reg.all_names() == []


class TestMissingRoot:
    def test_missing_root_raises_runtime_error(self, tmp_path: Path) -> None:
        cfg = NamespaceConfig(
            root="/this/path/does/not/exist/at/all", default_mode="rw"
        )
        fs_settings = FilesystemsSettings(namespaces={"session": cfg})

        class _FakeSettings:
            filesystems = fs_settings

        with pytest.raises(RuntimeError, match="does not exist"):
            NamespaceRegistry(settings=_FakeSettings())  # type: ignore[arg-type]

    def test_empty_root_skipped(self, tmp_path: Path) -> None:
        """A namespace with an empty root is silently skipped — not a startup error."""
        cfg = NamespaceConfig(root="", default_mode="rw")
        fs_settings = FilesystemsSettings(namespaces={"artifacts": cfg})

        class _FakeSettings:
            filesystems = fs_settings

        reg = NamespaceRegistry(settings=_FakeSettings())  # type: ignore[arg-type]
        assert "artifacts" not in reg.all_names()
