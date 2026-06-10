from __future__ import annotations

import time
from typing import TYPE_CHECKING

from private_gpt.components.code_execution.base import (
    BashExecutionResult,
    CodeExecutionSession,
    FileOperationResult,
)
from private_gpt.components.sandbox.base import SandboxExecOptions

if TYPE_CHECKING:
    from private_gpt.components.sandbox.base import AsyncSandboxSession


class SandboxCodeExecutionSession(CodeExecutionSession):
    """Generic CodeExecutionSession backed by any AsyncSandboxSession.

    The sandbox is responsible for path translation and permission enforcement.
    This class only adds the CodeExecutionSession protocol on top.
    """

    def __init__(
        self,
        session_id: str,
        sandbox: AsyncSandboxSession,
        workspace_canonical: str,
        last_accessed: list[float],
    ) -> None:
        self._id = session_id
        self._sandbox = sandbox
        self._workspace = workspace_canonical
        self._last_accessed = last_accessed

    def _touch(self) -> None:
        self._last_accessed[0] = time.monotonic()

    async def execute_bash(
        self, command: str, timeout: int | None = None, restart: bool = False
    ) -> BashExecutionResult:
        self._touch()
        if restart:
            await self._sandbox.exec(
                f"rm -rf {self._workspace}* {self._workspace}.[!.]*",
                SandboxExecOptions(cwd="/"),
            )
            await self._sandbox.make_dir(self._workspace)
        result = await self._sandbox.exec(
            command,
            SandboxExecOptions(timeout=timeout, cwd=self._workspace),
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
        self._touch()
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
        self._touch()
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
        self._touch()
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
        self._touch()
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

    async def close(self) -> None:
        await self._sandbox.close()
