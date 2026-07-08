from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class PathTranslator:
    """Stateless value object built once per session from its mount table.

    Maps LLM-visible canonical paths (e.g. /home/agent/) to real local paths
    and back. Mounts are sorted longest-canonical-prefix-first to avoid
    ambiguous prefix matching.

    All methods are pure (no I/O).
    """

    def __init__(self, mounts: list[tuple[str, Path, bool]]) -> None:
        # mounts: list of (canonical_prefix, real_path, writable)
        self._mounts = sorted(mounts, key=lambda m: len(m[0]), reverse=True)
        self._rebuild_regex()

    def _rebuild_regex(self) -> None:
        # Pre-compile a regex that matches any canonical prefix in a string.
        # Patterns are sorted longest-first so the leftmost-longest rule applies.
        escaped = [re.escape(canonical) for canonical, _, _ in self._mounts]
        if escaped:
            self._canonical_re = re.compile("|".join(escaped))
        else:
            self._canonical_re = re.compile(r"(?!)")  # never matches

        # Reverse: match any real path prefix.
        real_escaped = [re.escape(str(real)) + r"(/|$)" for _, real, _ in self._mounts]
        if real_escaped:
            self._real_re = re.compile("|".join(real_escaped))
        else:
            self._real_re = re.compile(r"(?!)")

    def register(self, canonical: str, real_path: Path, writable: bool) -> None:
        """Add or update a mount mapping and rebuild the internal regex."""
        self._mounts = [(c, r, w) for c, r, w in self._mounts if c != canonical]
        self._mounts.append((canonical, real_path, writable))
        self._mounts.sort(key=lambda m: len(m[0]), reverse=True)
        self._rebuild_regex()

    def unregister(self, canonical: str) -> None:
        """Remove a mount mapping and rebuild the internal regex."""
        self._mounts = [(c, r, w) for c, r, w in self._mounts if c != canonical]
        self._rebuild_regex()

    # ------------------------------------------------------------------
    # Path translation helpers
    # ------------------------------------------------------------------

    def to_real(self, canonical_path: str) -> Path:
        """Translate a canonical path to its real filesystem Path.

        Raises ValueError if the path does not start with any known mount prefix.
        """
        for canonical, real, _ in self._mounts:
            if canonical_path.startswith(canonical):
                relative = canonical_path[len(canonical) :]
                return real / relative
        raise ValueError(f"Path '{canonical_path}' does not match any session mount.")

    def to_canonical(self, real: Path | str) -> str:
        """Reverse-translate a real path to its canonical form.

        Raises ValueError if the real path is outside all mount points.
        """
        real_str = str(real)
        for canonical, mount_real, _ in self._mounts:
            mount_str = str(mount_real)
            if real_str == mount_str or real_str.startswith(mount_str + "/"):
                relative = real_str[len(mount_str) :]
                return canonical + relative.lstrip("/")
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
            for can, real, _ in self._mounts:
                if canonical == can:
                    return str(real)
            return canonical  # should never happen

        return self._canonical_re.sub(_replace, command)

    def scrub_output(self, output: str) -> str:
        """Replace all real mount paths in stdout/stderr with canonical paths."""
        if not self._mounts:
            return output

        result = output
        # Process longest real paths first (already sorted by canonical length desc,
        # which correlates with real path length).
        for canonical, real, _ in self._mounts:
            real_str = str(real)
            result = result.replace(real_str + "/", canonical)
            result = result.replace(real_str, canonical.rstrip("/"))
        return result
