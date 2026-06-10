from __future__ import annotations

import asyncio
import time
from datetime import timedelta
from typing import TYPE_CHECKING

from injector import inject

from private_gpt.components.code_execution.base import (
    BashExecutionResult,
    CodeExecutionProvider,
    CodeExecutionSession,
    FileOperationResult,
)
from private_gpt.components.skills.services.skill_loader import (
    SkillLoader,  # noqa: TC001
)
from private_gpt.settings.settings import Settings  # noqa: TC001

if TYPE_CHECKING:
    from opensandbox import Sandbox
    from opensandbox.models.execd import Execution

    from private_gpt.components.code_execution.content_bundle import ContentBundle
    from private_gpt.components.skills.models.skill_entities import SkillFilter


def _stdout_text(execution: Execution) -> str:
    return "\n".join(msg.text.rstrip("\n") for msg in execution.logs.stdout)


def _stderr_text(execution: Execution) -> str:
    return "\n".join(msg.text.rstrip("\n") for msg in execution.logs.stderr)


class OpenSandboxCodeExecutionSession(CodeExecutionSession):
    """Code execution session backed by an OpenSandbox container.

    The sandbox natively uses canonical paths (/home/agent/, /mnt/skills/, etc.),
    so no path translation is required. Skills are uploaded at session creation
    and marked read-only; the read-only enforcement is done at OS level inside
    the container (chmod 444).
    """

    def __init__(
        self,
        session_id: str,
        sandbox: Sandbox,
        readonly_prefixes: list[str],
    ) -> None:
        self._id = session_id
        self._sandbox = sandbox
        self._readonly_prefixes = readonly_prefixes
        self.last_accessed: float = time.monotonic()

    def _touch(self) -> None:
        self.last_accessed = time.monotonic()

    def _assert_writable(self, path: str) -> None:
        for prefix in self._readonly_prefixes:
            if path.startswith(prefix):
                raise ValueError(f"Path '{path}' is in a read-only mount ('{prefix}').")

    async def execute_bash(
        self, command: str, timeout: int | None = None, restart: bool = False
    ) -> BashExecutionResult:
        from opensandbox.models.execd import RunCommandOpts

        self._touch()
        if restart:
            await self._sandbox.commands.run(
                "rm -rf /home/agent/* /home/agent/.[!.]*",
                opts=RunCommandOpts(working_directory="/"),
            )

        t0 = time.monotonic()
        try:
            execution = await self._sandbox.commands.run(
                command,
                opts=RunCommandOpts(
                    working_directory="/home/agent/",
                    timeout=timedelta(seconds=timeout) if timeout else None,
                ),
            )
        except Exception as exc:
            return BashExecutionResult(
                success=False,
                stdout="",
                stderr=str(exc),
                exit_code=1,
                execution_time_ms=int((time.monotonic() - t0) * 1000),
            )

        exit_code = (
            execution.exit_code
            if execution.exit_code is not None
            else (0 if execution.error is None else 1)
        )
        return BashExecutionResult(
            success=(exit_code == 0),
            stdout=_stdout_text(execution),
            stderr=_stderr_text(execution),
            exit_code=exit_code,
            execution_time_ms=int((time.monotonic() - t0) * 1000),
        )

    async def view(
        self, path: str, view_range: tuple[int, int] | None = None
    ) -> FileOperationResult:
        from opensandbox.models.execd import RunCommandOpts

        self._touch()
        try:
            # Use ls to check if path is directory or file
            stat_result = await self._sandbox.commands.run(
                f"test -d {path!r} && echo dir || echo file",
                opts=RunCommandOpts(working_directory="/"),
            )
            entry_type = _stdout_text(stat_result).strip()

            if entry_type == "dir":
                ls_result = await self._sandbox.commands.run(
                    f"ls -1 {path!r}",
                    opts=RunCommandOpts(working_directory="/"),
                )
                entries = _stdout_text(ls_result).splitlines()
                lines = []
                for name in sorted(entries):
                    is_dir_result = await self._sandbox.commands.run(
                        f"test -d {path!r}/{name!r} && echo dir || echo file",
                        opts=RunCommandOpts(working_directory="/"),
                    )
                    tag = (
                        "[dir]"
                        if _stdout_text(is_dir_result).strip() == "dir"
                        else "[file]"
                    )
                    lines.append(f"{tag} {name}")
                return FileOperationResult(success=True, output="\n".join(lines))

            text = await self._sandbox.files.read_file(path)
            text_lines = text.splitlines()
            base_line = 1
            if view_range is not None:
                start, end = view_range
                start_idx = max(start, 1) - 1
                end_idx = None if end == -1 else max(end, 0)
                text_lines = text_lines[start_idx:end_idx]
                base_line = start_idx + 1
            output = "\n".join(
                f"{i}: {line}" for i, line in enumerate(text_lines, start=base_line)
            )
            return FileOperationResult(success=True, output=output)
        except Exception as exc:
            return FileOperationResult(success=False, error=str(exc))

    async def str_replace(
        self, path: str, old_str: str, new_str: str
    ) -> FileOperationResult:
        self._touch()
        try:
            self._assert_writable(path)
            text = await self._sandbox.files.read_file(path)
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
            await self._sandbox.files.write_file(path, updated)
            return FileOperationResult(success=True, output=f"Updated {path}")
        except Exception as exc:
            return FileOperationResult(success=False, error=str(exc))

    async def create(self, path: str, file_text: str) -> FileOperationResult:
        self._touch()
        try:
            self._assert_writable(path)
            from opensandbox.models.execd import RunCommandOpts

            # Check if file already exists
            check = await self._sandbox.commands.run(
                f"test -e {path!r} && echo exists || echo missing",
                opts=RunCommandOpts(working_directory="/"),
            )
            if _stdout_text(check).strip() == "exists":
                return FileOperationResult(
                    success=False, error=f"File already exists: {path}"
                )
            await self._sandbox.files.write_file(path, file_text)
            return FileOperationResult(success=True, output=f"Created {path}")
        except Exception as exc:
            return FileOperationResult(success=False, error=str(exc))

    async def insert(
        self, path: str, insert_line: int, new_str: str
    ) -> FileOperationResult:
        self._touch()
        try:
            self._assert_writable(path)
            text = await self._sandbox.files.read_file(path)
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
            await self._sandbox.files.write_file(path, updated)
            return FileOperationResult(success=True, output=f"Updated {path}")
        except Exception as exc:
            return FileOperationResult(success=False, error=str(exc))

    async def close(self) -> None:
        pass  # Provider owns sandbox lifecycle


class OpenSandboxCodeExecutionProvider(CodeExecutionProvider):
    """Code execution provider backed by an OpenSandbox server (Docker/K8s).

    Creates one sandbox container per session. Skills and other ContentBundles
    are uploaded into the container at their canonical paths and marked read-only.
    The sandbox natively uses canonical paths — no PathTranslator needed.
    """

    @inject
    def __init__(self, settings: Settings, skill_loader: SkillLoader) -> None:
        super().__init__(settings)
        self._cfg = settings.code_execution.opensandbox
        self._skill_loader = skill_loader
        self._active: dict[str, tuple[OpenSandboxCodeExecutionSession, Sandbox]] = {}
        self._lock = asyncio.Lock()
        self._reaper_started = False

    async def create_session(
        self,
        session_id: str,
        skill_filter: SkillFilter | None = None,
        extra_bundles: list[ContentBundle] | None = None,
    ) -> OpenSandboxCodeExecutionSession:
        from opensandbox import Sandbox
        from opensandbox.config import ConnectionConfig

        if self._cfg is None:
            raise RuntimeError(
                "OpenSandbox provider requires code_execution.opensandbox settings."
            )

        config = ConnectionConfig(
            domain=self._cfg.base_url,
            api_key=self._cfg.api_key,
        )
        sandbox = await Sandbox.create(
            self._cfg.image,
            connection_config=config,
            timeout=timedelta(seconds=self._cfg.session_ttl_seconds),
            resource=self._cfg.resource_limits,
        )

        # Upload content bundles into sandbox at their canonical paths
        readonly_prefixes: list[str] = []
        bundles: list[ContentBundle] = []
        if skill_filter:
            from private_gpt.components.code_execution.content_bundle import (
                ContentBundle as _CB,  # noqa: F401
            )

            bundles.extend(await self._skill_loader.load(skill_filter))
        if extra_bundles:
            bundles.extend(extra_bundles)

        for bundle in bundles:
            for rel_path, content in bundle.files.items():
                dest = bundle.canonical_path.rstrip("/") + "/" + rel_path
                await sandbox.files.write_file(dest, content)
            if not bundle.writable:
                await sandbox.commands.run(f"chmod -R 444 {bundle.canonical_path!r}")
                readonly_prefixes.append(bundle.canonical_path)

        session = OpenSandboxCodeExecutionSession(
            session_id, sandbox, readonly_prefixes
        )

        async with self._lock:
            self._active[session_id] = (session, sandbox)

        self._ensure_reaper()
        return session

    def delete_session(self, session: CodeExecutionSession) -> None:
        sid = getattr(session, "_id", None)
        if sid:
            entry = self._active.pop(sid, None)
            if entry:
                _, sandbox = entry
                asyncio.get_event_loop().create_task(sandbox.kill())

    def _ensure_reaper(self) -> None:
        if not self._reaper_started:
            self._reaper_started = True
            asyncio.get_event_loop().create_task(self._reaper_loop())

    async def _reaper_loop(self) -> None:
        if self._cfg is None:
            return
        ttl = self._cfg.session_ttl_seconds
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            async with self._lock:
                expired = [
                    (sid, sb)
                    for sid, (sess, sb) in self._active.items()
                    if now - sess.last_accessed > ttl
                ]
            for sid, sandbox in expired:
                async with self._lock:
                    self._active.pop(sid, None)
                asyncio.get_event_loop().create_task(sandbox.kill())
