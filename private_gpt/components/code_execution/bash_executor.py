from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import time
from asyncio.subprocess import PIPE
from typing import TYPE_CHECKING

from private_gpt.components.code_execution.base import BashExecutionResult

if TYPE_CHECKING:
    from pathlib import Path

_ISOLATION_AVAILABLE = sys.platform != "win32"


def _child_setup(cpu_s: int, mem_mb: int, fsize_mb: int, nproc: int) -> None:
    """Runs in child process before exec — sets up isolation via rlimit + setsid.

    setrlimit calls are best-effort: some platforms (macOS) reject certain
    limits (e.g. RLIMIT_AS always fails with a finite value).
    """
    os.setsid()
    import resource as r

    def _set(which: int, limit: tuple[int, int]) -> None:
        with contextlib.suppress(ValueError, OSError):
            r.setrlimit(which, limit)

    _set(r.RLIMIT_CPU, (cpu_s, cpu_s))
    _set(r.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
    _set(r.RLIMIT_FSIZE, (fsize_mb << 20, fsize_mb << 20))
    _set(r.RLIMIT_NPROC, (nproc, nproc))


def _cap_bytes(data: bytes, limit: int) -> str:
    if len(data) > limit:
        truncated = data[:limit]
        return (
            truncated.decode("utf-8", errors="replace")
            + f"\n[output truncated at {limit} bytes]"
        )
    return data.decode("utf-8", errors="replace")


class LocalBashExecutor:
    """Async subprocess executor with resource isolation (Unix) and output capping.

    On Unix: uses os.setsid + setrlimit (CPU, virtual memory, file size, nproc).
    On Windows: runs without isolation (same behavior as before).
    Timeout kills the entire process group via SIGKILL.
    """

    def __init__(
        self,
        cpu_limit_seconds: int = 30,
        memory_limit_mb: int = 512,
        fsize_limit_mb: int = 50,
        nproc_limit: int = 50,
        output_cap_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        self._cpu_s = cpu_limit_seconds
        self._mem_mb = memory_limit_mb
        self._fsize_mb = fsize_limit_mb
        self._nproc = nproc_limit
        self._cap = output_cap_bytes

    async def run(
        self,
        command: str,
        cwd: Path,
        timeout: int | None = None,
    ) -> BashExecutionResult:
        import functools

        preexec = (
            functools.partial(
                _child_setup, self._cpu_s, self._mem_mb, self._fsize_mb, self._nproc
            )
            if _ISOLATION_AVAILABLE
            else None
        )

        t0 = time.monotonic()
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=PIPE,
            stderr=PIPE,
            cwd=str(cwd),
            preexec_fn=preexec,
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(), timeout=float(timeout) if timeout else 60.0
            )
        except TimeoutError:
            try:
                if _ISOLATION_AVAILABLE:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            return BashExecutionResult(
                success=False,
                stdout="",
                stderr=f"Command timed out after {timeout}s.",
                exit_code=124,
                execution_time_ms=int((time.monotonic() - t0) * 1000),
            )

        return BashExecutionResult(
            success=(proc.returncode == 0),
            stdout=_cap_bytes(out, self._cap),
            stderr=_cap_bytes(err, self._cap),
            exit_code=proc.returncode or 0,
            execution_time_ms=int((time.monotonic() - t0) * 1000),
        )
