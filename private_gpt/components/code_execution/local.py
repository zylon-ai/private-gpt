import shutil
from pathlib import Path

import anyio

from private_gpt.components.code_execution.base import (
    BashExecutionResult,
    CodeExecutionProvider,
    CodeExecutionSession,
    FileOperationResult,
)
from private_gpt.components.sandbox.base import SandboxExecOptions
from private_gpt.components.sandbox.local import LocalSandboxSession
from private_gpt.components.sandbox.registry import SandboxProviderRegistry
from private_gpt.settings.settings import Settings


class LocalCodeExecutionSession(CodeExecutionSession):
    def __init__(self, sandbox_session: LocalSandboxSession, workspace: Path) -> None:
        self._sandbox = sandbox_session
        self._workspace = workspace.resolve()
        self._workspace.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, user_path: str) -> Path:
        cleaned_path = user_path.lstrip("/")
        candidate = (self._workspace / cleaned_path).resolve()
        if candidate != self._workspace and self._workspace not in candidate.parents:
            raise ValueError(f"Path '{user_path}' escapes the session workspace.")
        return candidate

    async def execute_bash(
        self, command: str, timeout: int | None = None, restart: bool = False
    ) -> BashExecutionResult:
        if restart:
            await anyio.to_thread.run_sync(
                lambda: shutil.rmtree(self._workspace, ignore_errors=True)
            )
            await anyio.to_thread.run_sync(
                lambda: self._workspace.mkdir(parents=True, exist_ok=True)
            )

        result = await anyio.to_thread.run_sync(
            lambda: self._sandbox.exec(
                command,
                SandboxExecOptions(
                    cwd=str(self._workspace),
                    timeout=timeout,
                ),
            )
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
        try:
            target = self._resolve_path(path)
            if not await anyio.to_thread.run_sync(target.exists):
                return FileOperationResult(
                    success=False,
                    error=f"File not found: {path}",
                )

            if await anyio.to_thread.run_sync(target.is_dir):
                entries = await anyio.to_thread.run_sync(
                    lambda: [
                        (entry.name, entry.is_dir())
                        for entry in sorted(
                            target.iterdir(), key=lambda item: item.name
                        )
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
                f"{index}: {line}" for index, line in enumerate(lines, start=base_line)
            )
            return FileOperationResult(success=True, output=output)
        except Exception as exc:
            return FileOperationResult(success=False, error=str(exc))

    async def str_replace(
        self, path: str, old_str: str, new_str: str
    ) -> FileOperationResult:
        try:
            target = self._resolve_path(path)
            if not await anyio.to_thread.run_sync(target.exists):
                return FileOperationResult(
                    success=False,
                    error=f"File not found: {path}",
                )

            text = await anyio.to_thread.run_sync(
                lambda: target.read_text(encoding="utf-8")
            )
            occurrences = text.count(old_str)
            if occurrences == 0:
                return FileOperationResult(
                    success=False,
                    error="old_str was not found in the file.",
                )
            if occurrences > 1:
                return FileOperationResult(
                    success=False,
                    error="old_str appears more than once in the file.",
                )

            updated = text.replace(old_str, new_str, 1)
            await anyio.to_thread.run_sync(
                lambda: target.write_text(updated, encoding="utf-8")
            )
            return FileOperationResult(success=True, output=f"Updated {path}")
        except Exception as exc:
            return FileOperationResult(success=False, error=str(exc))

    async def create(self, path: str, file_text: str) -> FileOperationResult:
        try:
            target = self._resolve_path(path)
            if await anyio.to_thread.run_sync(target.exists):
                return FileOperationResult(
                    success=False,
                    error=f"File already exists: {path}",
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
        try:
            target = self._resolve_path(path)
            if not await anyio.to_thread.run_sync(target.exists):
                return FileOperationResult(
                    success=False,
                    error=f"File not found: {path}",
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
        await anyio.to_thread.run_sync(self._sandbox.close)

    @property
    def sandbox_session(self) -> LocalSandboxSession:
        return self._sandbox

    @property
    def workspace(self) -> Path:
        return self._workspace


class LocalCodeExecutionProvider(CodeExecutionProvider):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._sandbox_provider = SandboxProviderRegistry(settings).get_provider("local")
        self._base_path = Path(
            settings.code_execution.workspace_path
            or Path(settings.data.local_data_folder) / "code_execution_workspaces"
        )

    def create_session(self, session_id: str) -> CodeExecutionSession:
        sandbox_session = self._sandbox_provider.create_session(
            user_id=session_id,
            timeout=self.settings.code_execution.timeout,
        )
        if not isinstance(sandbox_session, LocalSandboxSession):
            raise TypeError("Local code execution requires a LocalSandboxSession.")
        sandbox_session.start()
        workspace = self._base_path / session_id
        return LocalCodeExecutionSession(
            sandbox_session=sandbox_session,
            workspace=workspace,
        )

    def delete_session(self, session: CodeExecutionSession) -> None:
        if not isinstance(session, LocalCodeExecutionSession):
            raise TypeError(
                "Local code execution requires a LocalCodeExecutionSession."
            )
        self._sandbox_provider.delete_session(session.sandbox_session)
        shutil.rmtree(session.workspace, ignore_errors=True)
