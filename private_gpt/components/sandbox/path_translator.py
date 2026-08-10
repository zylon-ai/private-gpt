from __future__ import annotations

import re
from typing import TYPE_CHECKING

from private_gpt.components.sandbox.mount import Mount

if TYPE_CHECKING:
    from pathlib import Path


class PathTranslator:
    """Stateless value object built once per session from its mount table.

    Maps LLM-visible canonical paths (e.g. /home/agent/) to real local paths
    and back. Mounts are sorted longest-canonical-prefix-first to avoid
    ambiguous prefix matching.

    All methods are pure (no I/O).
    """

    def __init__(self, mounts: list[Mount]) -> None:
        self._mounts = sorted(
            (m for m in mounts if m.host_path is not None),
            key=lambda m: len(m.target),
            reverse=True,
        )
        self._rebuild_regex()

    def _rebuild_regex(self) -> None:
        escaped = [re.escape(m.target) for m in self._mounts]
        if escaped:
            self._canonical_re = re.compile("|".join(escaped))
        else:
            self._canonical_re = re.compile(r"(?!)")  # never matches

        real_escaped = [re.escape(str(m.host_path)) + r"(/|$)" for m in self._mounts]
        if real_escaped:
            self._real_re = re.compile("|".join(real_escaped))
        else:
            self._real_re = re.compile(r"(?!)")

    def register(self, canonical: str, host_path: Path, writable: bool) -> None:
        """Add or update a mount mapping and rebuild the internal regex."""
        self._mounts = [m for m in self._mounts if m.target != canonical]
        self._mounts.append(
            Mount(
                target=canonical, access="rw" if writable else "ro", host_path=host_path
            )
        )
        self._mounts.sort(key=lambda m: len(m.target), reverse=True)
        self._rebuild_regex()

    def unregister(self, canonical: str) -> None:
        """Remove a mount mapping and rebuild the internal regex."""
        self._mounts = [m for m in self._mounts if m.target != canonical]
        self._rebuild_regex()

    # ------------------------------------------------------------------
    # Path translation helpers
    # ------------------------------------------------------------------

    def to_real(self, canonical_path: str) -> Path:
        """Translate a canonical path to its real filesystem Path.

        Folder mounts match by prefix; file mounts match exactly (a file mount
        maps one canonical file to one host file, never a subtree).
        Raises ValueError if the path does not match any known mount.
        """
        for mount in self._mounts:
            if mount.target.endswith("/"):
                if canonical_path.startswith(mount.target):
                    relative = canonical_path[len(mount.target) :]
                    assert mount.host_path is not None  # filtered in __init__
                    return mount.host_path / relative
            elif canonical_path == mount.target:
                assert mount.host_path is not None  # filtered in __init__
                return mount.host_path
        raise ValueError(f"Path '{canonical_path}' does not match any session mount.")

    def to_canonical(self, real: Path | str) -> str:
        """Reverse-translate a real path to its canonical form.

        Folder mounts match by prefix; file mounts match exactly.
        Raises ValueError if the real path is outside all mount points.
        """
        real_str = str(real)
        for mount in self._mounts:
            assert mount.host_path is not None  # filtered in __init__
            mount_str = str(mount.host_path)
            if mount.target.endswith("/"):
                if real_str == mount_str or real_str.startswith(mount_str + "/"):
                    relative = real_str[len(mount_str) :]
                    return mount.target + relative.lstrip("/")
            elif real_str == mount_str:
                return mount.target
        raise ValueError(f"Real path '{real}' is not inside any session mount.")

    # ------------------------------------------------------------------
    # String rewriting (commands and output)
    # ------------------------------------------------------------------

    def rewrite_command(self, command: str) -> str:
        """Replace all canonical path prefixes in a command string with real paths."""
        if not self._mounts:
            return command

        def _replace(match: re.Match[str]) -> str:
            canonical = match.group(0)
            for mount in self._mounts:
                if canonical == mount.target:
                    assert mount.host_path is not None  # filtered in __init__
                    host = str(mount.host_path)
                    return host.rstrip("/") + "/" if canonical.endswith("/") else host
            return canonical  # should never happen

        return self._canonical_re.sub(_replace, command)

    def scrub_output(self, output: str) -> str:
        """Replace all real mount paths in stdout/stderr with canonical paths."""
        if not self._mounts:
            return output

        result = output
        for mount in self._mounts:
            assert mount.host_path is not None  # filtered in __init__
            real_str = str(mount.host_path)
            result = result.replace(real_str + "/", mount.target)
            result = result.replace(real_str, mount.target.rstrip("/"))
        return result
