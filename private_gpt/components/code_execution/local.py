from __future__ import annotations

import asyncio
import hashlib
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
from injector import inject

from private_gpt.components.code_execution.base import (
    BashExecutionResult,
    CodeExecutionProvider,
    CodeExecutionSession,
    FileOperationResult,
)
from private_gpt.components.code_execution.bash_executor import BashExecutor
from private_gpt.components.code_execution.mount import LocalMount, ReadOnlyMount
from private_gpt.components.code_execution.path_translator import PathTranslator
from private_gpt.settings.settings import Settings  # noqa: TC001

if TYPE_CHECKING:
    from private_gpt.components.code_execution.content_bundle import ContentBundle
    from private_gpt.components.code_execution.mount import SessionMount


def _cache_key(canonical_path: str) -> str:
    """Derive a short, filesystem-safe cache directory name from a canonical path."""
    return hashlib.sha1(canonical_path.encode()).hexdigest()[:16]


class LocalCodeExecutionSession(CodeExecutionSession):
    """Code execution session backed by the local filesystem.

    All path operations go through PathTranslator:
      /home/agent/     → {base}/sessions/{id}/workspace/   (writable)
      /mnt/user-data/  → {base}/sessions/{id}/uploads|outputs/
      /mnt/skills/...  → {base}/content_cache/{hash}/      (read-only)

    Falls back to workspace-relative resolution for bare/relative paths so
    that existing callers remain compatible.
    """

    def __init__(
        self,
        session_id: str,
        translator: PathTranslator,
        workspace: Path,
        last_accessed: list[float],
        executor: BashExecutor,
    ) -> None:
        self._id = session_id
        self._translator = translator
        self._workspace = workspace
        self._last_accessed = last_accessed  # mutable cell shared with provider
        self._executor = executor

    def _touch(self) -> None:
        self._last_accessed[0] = time.monotonic()

    def _resolve(self, path: str) -> Path:
        try:
            return self._translator.to_real(path)
        except ValueError:
            cleaned = path.lstrip("/")
            candidate = (self._workspace / cleaned).resolve()
            if (
                candidate != self._workspace
                and self._workspace not in candidate.parents
            ):
                raise ValueError(
                    f"Path '{path}' escapes the session workspace."
                ) from None
            return candidate

    async def execute_bash(
        self, command: str, timeout: int | None = None, restart: bool = False
    ) -> BashExecutionResult:
        self._touch()
        if restart:
            await anyio.to_thread.run_sync(
                lambda: shutil.rmtree(self._workspace, ignore_errors=True)
            )
            await anyio.to_thread.run_sync(
                lambda: self._workspace.mkdir(parents=True, exist_ok=True)
            )
        translated = self._translator.rewrite_command(command)
        result = await self._executor.run(
            translated, cwd=self._workspace, timeout=timeout
        )
        return BashExecutionResult(
            success=result.success,
            stdout=self._translator.scrub_output(result.stdout),
            stderr=self._translator.scrub_output(result.stderr),
            exit_code=result.exit_code,
            execution_time_ms=result.execution_time_ms,
        )

    async def view(
        self, path: str, view_range: tuple[int, int] | None = None
    ) -> FileOperationResult:
        self._touch()
        try:
            target = self._resolve(path)
            if not await anyio.to_thread.run_sync(target.exists):
                return FileOperationResult(
                    success=False, error=f"File not found: {path}"
                )

            if await anyio.to_thread.run_sync(target.is_dir):
                entries = await anyio.to_thread.run_sync(
                    lambda: [
                        (entry.name, entry.is_dir())
                        for entry in sorted(target.iterdir(), key=lambda e: e.name)
                    ]
                )
                lines = [
                    f"[dir] {name}" if is_dir else f"[file] {name}"
                    for name, is_dir in entries
                ]
                return FileOperationResult(success=True, output="\n".join(lines))

            text = await anyio.to_thread.run_sync(
                lambda: target.read_text(encoding="utf-8")
            )
            lines = text.splitlines()
            base_line = 1
            if view_range is not None:
                start, end = view_range
                start_index = max(start, 1) - 1
                end_index = None if end == -1 else max(end, 0)
                lines = lines[start_index:end_index]
                base_line = start_index + 1
            output = "\n".join(
                f"{i}: {line}" for i, line in enumerate(lines, start=base_line)
            )
            return FileOperationResult(success=True, output=output)
        except Exception as exc:
            return FileOperationResult(success=False, error=str(exc))

    async def str_replace(
        self, path: str, old_str: str, new_str: str
    ) -> FileOperationResult:
        self._touch()
        try:
            self._translator.assert_writable(path)
            target = self._resolve(path)
            if not await anyio.to_thread.run_sync(target.exists):
                return FileOperationResult(
                    success=False, error=f"File not found: {path}"
                )
            text = await anyio.to_thread.run_sync(
                lambda: target.read_text(encoding="utf-8")
            )
            occurrences = text.count(old_str)
            if occurrences == 0:
                return FileOperationResult(
                    success=False, error="old_str was not found in the file."
                )
            if occurrences > 1:
                return FileOperationResult(
                    success=False, error="old_str appears more than once in the file."
                )
            updated = text.replace(old_str, new_str, 1)
            await anyio.to_thread.run_sync(
                lambda: target.write_text(updated, encoding="utf-8")
            )
            return FileOperationResult(success=True, output=f"Updated {path}")
        except Exception as exc:
            return FileOperationResult(success=False, error=str(exc))

    async def create(self, path: str, file_text: str) -> FileOperationResult:
        self._touch()
        try:
            self._translator.assert_writable(path)
            target = self._resolve(path)
            if await anyio.to_thread.run_sync(target.exists):
                return FileOperationResult(
                    success=False, error=f"File already exists: {path}"
                )
            await anyio.to_thread.run_sync(
                lambda: target.parent.mkdir(parents=True, exist_ok=True)
            )
            await anyio.to_thread.run_sync(
                lambda: target.write_text(file_text, encoding="utf-8")
            )
            return FileOperationResult(success=True, output=f"Created {path}")
        except Exception as exc:
            return FileOperationResult(success=False, error=str(exc))

    async def insert(
        self, path: str, insert_line: int, new_str: str
    ) -> FileOperationResult:
        self._touch()
        try:
            self._translator.assert_writable(path)
            target = self._resolve(path)
            if not await anyio.to_thread.run_sync(target.exists):
                return FileOperationResult(
                    success=False, error=f"File not found: {path}"
                )
            text = await anyio.to_thread.run_sync(
                lambda: target.read_text(encoding="utf-8")
            )
            lines = text.splitlines()
            if insert_line < 0 or insert_line > len(lines):
                return FileOperationResult(
                    success=False,
                    error=f"insert_line {insert_line} is out of range.",
                )
            insertion = new_str.splitlines()
            updated_lines = lines[:insert_line] + insertion + lines[insert_line:]
            updated = "\n".join(updated_lines)
            if text.endswith("\n") or new_str.endswith("\n"):
                updated += "\n"
            await anyio.to_thread.run_sync(
                lambda: target.write_text(updated, encoding="utf-8")
            )
            return FileOperationResult(success=True, output=f"Updated {path}")
        except Exception as exc:
            return FileOperationResult(success=False, error=str(exc))

    async def close(self) -> None:
        pass  # Provider owns lifecycle; disk files preserved for kernel-restart recovery


class LocalCodeExecutionProvider(CodeExecutionProvider):
    """Provides LocalCodeExecutionSession instances backed by the local filesystem.

    Session directories survive kernel restarts — if a session directory already
    exists on disk when create_session() is called, the existing files are reused.

    Skills and other ContentBundles are materialised to a shared content cache so
    that the same version is not re-downloaded for every session.
    """

    @inject
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        ce = settings.code_execution
        base = Path(
            ce.workspace_path
            or Path(settings.data.local_data_folder) / "code_execution_workspaces"
        )
        self._sessions_dir = base / "sessions"
        self._content_cache = Path(ce.skills_cache_path or base / "content_cache")
        self._executor = BashExecutor(
            cpu_limit_seconds=ce.bash_cpu_limit_seconds,
            memory_limit_mb=ce.bash_memory_limit_mb,
            fsize_limit_mb=ce.bash_fsize_limit_mb,
            nproc_limit=ce.bash_nproc_limit,
            output_cap_bytes=ce.bash_output_cap_bytes,
        )
        self._ttl = ce.session_ttl_seconds
        self._active: dict[str, tuple[LocalCodeExecutionSession, list[float]]] = {}
        self._lock = asyncio.Lock()
        self._reaper_started = False

    async def create_session(
        self,
        session_id: str,
        extra_bundles: list[ContentBundle] | None = None,
    ) -> LocalCodeExecutionSession:
        session_dir = self._sessions_dir / session_id

        mounts: list[SessionMount] = [
            LocalMount("/home/agent/", session_dir / "workspace", writable=True),
            LocalMount(
                "/mnt/user-data/uploads/", session_dir / "uploads", writable=False
            ),
            LocalMount(
                "/mnt/user-data/outputs/", session_dir / "outputs", writable=True
            ),
        ]

        for bundle in extra_bundles or []:
            cache_dir = self._content_cache / _cache_key(bundle.canonical_path)
            mounts.append(ReadOnlyMount(bundle.canonical_path, bundle.files, cache_dir))

        real_paths = await asyncio.gather(*[m.prepare() for m in mounts])
        translator = PathTranslator(
            [
                (m.canonical, rp, m.writable)
                for m, rp in zip(mounts, real_paths, strict=True)
            ]
        )
        workspace = real_paths[0]
        last_accessed: list[float] = [time.monotonic()]
        session = LocalCodeExecutionSession(
            session_id, translator, workspace, last_accessed, self._executor
        )

        async with self._lock:
            self._active[session_id] = (session, last_accessed)

        self._ensure_reaper()
        return session

    def delete_session(self, session: CodeExecutionSession) -> None:
        if isinstance(session, LocalCodeExecutionSession):
            sid = session._id
            self._active.pop(sid, None)
            # Disk files preserved — next create_session() with the same session_id
            # will reuse the existing workspace directory.

    def _ensure_reaper(self) -> None:
        if not self._reaper_started:
            self._reaper_started = True
            asyncio.get_event_loop().create_task(self._reaper_loop())

    async def _reaper_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            async with self._lock:
                expired = [
                    sid
                    for sid, (_, last) in self._active.items()
                    if now - last[0] > self._ttl
                ]
            for sid in expired:
                async with self._lock:
                    self._active.pop(sid, None)
