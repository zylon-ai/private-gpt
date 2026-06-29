from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import anyio.to_thread

from private_gpt.components.code_execution.bash_executor import LocalBashExecutor
from private_gpt.components.code_execution.path_translator import (
    PathTranslator,
)
from private_gpt.components.sandbox.base import (
    SandboxExecutionResult,
    SandboxProvider,
    SandboxSession,
)
from private_gpt.components.sandbox.mount import LocalMountSpec

if TYPE_CHECKING:
    from private_gpt.components.sandbox.base import (
        SandboxExecOptions,
    )
    from private_gpt.components.sandbox.content_bundle import BundledFile
    from private_gpt.components.sandbox.mount import SandboxMountSpec, VolumeSpec
    from private_gpt.settings.settings import Settings


class BashExecutorSandbox(SandboxSession):
    """Local async sandbox: BashExecutor for exec, pathlib for file ops.

    Translates canonical paths → real local paths via PathTranslator.
    Enforces read-only constraints on write_file() and chmod().
    initialize_mount() bypasses the check for session setup.
    """

    python_executable: str = sys.executable

    def __init__(
        self, mounts: list[LocalMountSpec], executor: LocalBashExecutor
    ) -> None:
        self._translator = PathTranslator(
            [(m.canonical, m.real_path, m.writable) for m in mounts]
        )
        self._executor = executor
        self._readonly = [m.canonical for m in mounts if not m.writable]
        self._default_cwd = next((m.canonical for m in mounts if m.writable), "/")

    def _assert_writable(self, path: str) -> None:
        for prefix in self._readonly:
            if path.startswith(prefix):
                raise ValueError(f"Path '{path}' is in a read-only mount ('{prefix}').")

    async def exec(
        self, command: str, opts: SandboxExecOptions | None = None
    ) -> SandboxExecutionResult:
        cwd = self._translator.to_real(
            (opts.cwd if opts else None) or self._default_cwd
        )
        cmd = self._translator.rewrite_command(command)
        result = await self._executor.run(
            cmd, cwd=cwd, timeout=opts.timeout if opts else None
        )
        return SandboxExecutionResult(
            success=result.success,
            stdout=self._translator.scrub_output(result.stdout),
            stderr=self._translator.scrub_output(result.stderr),
            exit_code=result.exit_code,
            execution_time_ms=result.execution_time_ms,
        )

    async def read_file(self, path: str) -> bytes:
        real = self._translator.to_real(path)
        return await anyio.to_thread.run_sync(real.read_bytes)

    async def write_file(self, path: str, content: bytes) -> None:
        self._assert_writable(path)
        real = self._translator.to_real(path)
        await anyio.to_thread.run_sync(
            lambda: real.parent.mkdir(parents=True, exist_ok=True)
        )
        await anyio.to_thread.run_sync(lambda: real.write_bytes(content))

    async def path_exists(self, path: str) -> bool:
        real = self._translator.to_real(path)
        return await anyio.to_thread.run_sync(real.exists)

    async def is_dir(self, path: str) -> bool:
        real = self._translator.to_real(path)
        return await anyio.to_thread.run_sync(real.is_dir)

    async def list_dir(self, path: str) -> list[str]:
        real = self._translator.to_real(path)
        entries: list[tuple[str, bool]] = await anyio.to_thread.run_sync(
            lambda: [
                (e.name, e.is_dir())
                for e in sorted(real.iterdir(), key=lambda e: e.name)
            ]
        )
        return [f"[dir] {name}" if is_d else f"[file] {name}" for name, is_d in entries]

    async def make_dir(
        self, path: str, *, parents: bool = True, exist_ok: bool = True
    ) -> None:
        real = self._translator.to_real(path)
        await anyio.to_thread.run_sync(
            lambda: real.mkdir(parents=parents, exist_ok=exist_ok)
        )

    async def chmod(self, path: str, mode: int) -> None:
        self._assert_writable(path)
        real = self._translator.to_real(path)
        await anyio.to_thread.run_sync(lambda: real.chmod(mode))

    async def initialize_mount(self, canonical: str, files: list[BundledFile]) -> None:
        for f in files:
            real = self._translator.to_real(canonical + f.path)
            content = f.content
            permissions = f.permissions

            def _mkdir(r: Path = real) -> None:
                r.parent.mkdir(parents=True, exist_ok=True)

            def _write(r: Path = real, c: bytes = content) -> None:
                r.write_bytes(c)

            def _chmod(r: Path = real, p: int = permissions) -> None:
                r.chmod(p)

            await anyio.to_thread.run_sync(_mkdir)
            await anyio.to_thread.run_sync(_write)
            await anyio.to_thread.run_sync(_chmod)

    async def close(self) -> None:
        pass


class LocalSandboxSession(BashExecutorSandbox):
    """BashExecutorSandbox that owns its temporary workspace directory."""

    def __init__(
        self,
        mounts: list[LocalMountSpec],
        executor: LocalBashExecutor,
        workdir: Path,
    ) -> None:
        super().__init__(mounts, executor)
        self._workdir = workdir

    async def close(self) -> None:
        await anyio.to_thread.run_sync(
            lambda: shutil.rmtree(self._workdir, ignore_errors=True)
        )


class LocalSandboxProvider(SandboxProvider):
    """Local sandbox provider for development and OSS defaults."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._executor = LocalBashExecutor(
            cpu_limit_seconds=settings.bash.cpu_limit_seconds,
            memory_limit_mb=settings.bash.memory_limit_mb,
            fsize_limit_mb=settings.bash.fsize_limit_mb,
            nproc_limit=settings.bash.nproc_limit,
            output_cap_bytes=settings.bash.output_cap_bytes,
        )

    async def create_session(
        self,
        user_id: str | None = None,
        timeout: int | None = None,
        bundle_specs: list[SandboxMountSpec] | None = None,
        *,
        session_id: str | None = None,
        volumes: list[VolumeSpec] | None = None,
    ) -> SandboxSession:
        if volumes:
            # Host-backed session: the mounter owns the directories, the
            # sandbox only translates canonical paths onto them.
            specs = [
                LocalMountSpec(
                    canonical=v.mount_path,
                    real_path=v.host_path,
                    writable=not v.read_only,
                )
                for v in volumes
            ]
            return BashExecutorSandbox(specs, self._executor)

        # Standalone use: an ephemeral workspace owned by the session.
        workdir = await anyio.to_thread.run_sync(
            lambda: Path(tempfile.mkdtemp(prefix=f"sandbox_{user_id or 'local'}_"))
        )
        specs = [
            LocalMountSpec(canonical="/home/agent/", real_path=workdir, writable=True)
        ]
        specs.extend(s for s in bundle_specs or [] if isinstance(s, LocalMountSpec))
        return LocalSandboxSession(specs, self._executor, workdir)

    async def delete_session(self, session: SandboxSession) -> None:
        if isinstance(session, BashExecutorSandbox):
            await session.close()
        else:
            raise TypeError("Expected a BashExecutorSandbox session instance.")
