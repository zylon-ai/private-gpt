from __future__ import annotations

import posixpath
from typing import TYPE_CHECKING

from private_gpt.components.code_execution.base import (
    BashExecutionResult,
    CodeExecutionSession,
    FileOperationResult,
)
from private_gpt.components.sandbox.base import SandboxExecOptions

if TYPE_CHECKING:
    from private_gpt.components.environment.environment import Environment
    from private_gpt.components.sandbox.base import SandboxSession


class SandboxCodeExecutionSession(CodeExecutionSession):
    """CodeExecutionSession tool protocol over a managed Environment.

    The environment owns lifetime/idle tracking; its sandbox owns path
    translation and permission enforcement. This class only adapts the
    tool protocol on top.
    """

    def __init__(self, environment: Environment) -> None:
        self._env = environment
        self._id = environment.id

    @property
    def _sandbox(self) -> SandboxSession:
        return self._env.sandbox

    def _resolve_path(self, path: str) -> str:
        if posixpath.isabs(path):
            return path
        return posixpath.join(self._env.workspace, path)

    async def execute_bash(
        self, command: str, timeout: int | None = None, restart: bool = False
    ) -> BashExecutionResult:
        workspace = self._env.workspace
        if restart:
            # No cwd: the default is the workspace itself, which every backend
            # can resolve (cwd="/" is outside the local translator's mounts).
            await self._env.exec(f"rm -rf {workspace}* {workspace}.[!.]*")
            await self._sandbox.make_dir(workspace)
        result = await self._env.exec(
            command,
            SandboxExecOptions(timeout=timeout, cwd=workspace),
        )
        return BashExecutionResult(
            success=result.success,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            execution_time_ms=result.execution_time_ms,
        )

    async def view(
        self, path: str, view_range: tuple[int, int] | None = None
    ) -> FileOperationResult:
        path = self._resolve_path(path)
        self._env.touch()
        try:
            if not await self._sandbox.path_exists(path):
                return FileOperationResult(
                    success=False, error=f"File not found: {path}"
                )
            if await self._sandbox.is_dir(path):
                entries = await self._sandbox.list_dir(path)
                return FileOperationResult(success=True, output="\n".join(entries))
            raw = await self._sandbox.read_file(path)
            text = raw.decode("utf-8", errors="replace")
            lines = text.splitlines()
            base_line = 1
            if view_range is not None:
                start, end = view_range
                start_idx = max(start, 1) - 1
                end_idx = None if end == -1 else max(end, 0)
                lines = lines[start_idx:end_idx]
                base_line = start_idx + 1
            output = "\n".join(
                f"{i}: {line}" for i, line in enumerate(lines, start=base_line)
            )
            return FileOperationResult(success=True, output=output)
        except Exception as exc:
            return FileOperationResult(success=False, error=str(exc))

    async def str_replace(
        self, path: str, old_str: str, new_str: str
    ) -> FileOperationResult:
        path = self._resolve_path(path)
        self._env.touch()
        try:
            raw = await self._sandbox.read_file(path)
            text = raw.decode("utf-8", errors="replace")
            occurrences = text.count(old_str)
            if occurrences == 0:
                return FileOperationResult(
                    success=False, error="old_str was not found in the file."
                )
            if occurrences > 1:
                return FileOperationResult(
                    success=False,
                    error="old_str appears more than once in the file.",
                )
            updated = text.replace(old_str, new_str, 1)
            await self._sandbox.write_file(path, updated.encode("utf-8"))
            return FileOperationResult(success=True, output=f"Updated {path}")
        except Exception as exc:
            return FileOperationResult(success=False, error=str(exc))

    async def create(self, path: str, file_text: str) -> FileOperationResult:
        path = self._resolve_path(path)
        self._env.touch()
        try:
            if await self._sandbox.path_exists(path):
                return FileOperationResult(
                    success=False, error=f"File already exists: {path}"
                )
            await self._sandbox.write_file(path, file_text.encode("utf-8"))
            return FileOperationResult(success=True, output=f"Created {path}")
        except Exception as exc:
            return FileOperationResult(success=False, error=str(exc))

    async def insert(
        self, path: str, insert_line: int, new_str: str
    ) -> FileOperationResult:
        path = self._resolve_path(path)
        self._env.touch()
        try:
            raw = await self._sandbox.read_file(path)
            text = raw.decode("utf-8", errors="replace")
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
            await self._sandbox.write_file(path, updated.encode("utf-8"))
            return FileOperationResult(success=True, output=f"Updated {path}")
        except Exception as exc:
            return FileOperationResult(success=False, error=str(exc))

    async def read_file(self, path: str) -> bytes:
        path = self._resolve_path(path)
        self._env.touch()
        return await self._sandbox.read_file(path)

    async def close(self) -> None:
        await self._sandbox.close()
