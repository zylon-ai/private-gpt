from collections.abc import Awaitable, Callable
from pathlib import PurePosixPath


async def unique_name(
    desired: str,
    exists: Callable[[str], Awaitable[bool]],
    max_attempts: int = 1000,
) -> str:
    """Return ``desired``, or ``stem (N).ext`` for the first free ``N >= 2``.

    Mirrors the Windows copy-suffix convention: only the last extension is kept
    apart, so ``report.tar.gz`` becomes ``report.tar (2).gz``.

    The check is not atomic — a concurrent writer may claim the name between the
    ``exists`` probe and the write that follows. Both call sites are
    single-writer per session, and the worst case is an overwrite rather than
    corruption, so no locking is used.
    """
    if not await exists(desired):
        return desired

    path = PurePosixPath(desired)
    stem, suffix = path.stem, path.suffix
    parent = str(path.parent) if path.parent != PurePosixPath(".") else ""

    for counter in range(2, max_attempts + 2):
        candidate = f"{stem} ({counter}){suffix}"
        if parent:
            candidate = f"{parent}/{candidate}"
        if not await exists(candidate):
            return candidate

    raise ValueError(
        f"Could not find a free name for {desired!r} after {max_attempts} attempts."
    )
