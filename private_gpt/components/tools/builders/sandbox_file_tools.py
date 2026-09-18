"""Shared scaffolding for tools that read a sandbox file and leave a markdown file.

``convert_documents``, ``describe_image`` and ``transcribe_audio`` differ only in
how they turn bytes into text. Everything around that — validating the path,
running a batch without letting one bad path fail its siblings, claiming a free
name in the workspace — is the same, and lives here.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from private_gpt.components.environment.layout import DEFAULT_SESSION_LAYOUT
from private_gpt.components.environment.naming import unique_name
from private_gpt.events.models import TextBlock

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from llama_index.core.llms import LLM

    from private_gpt.components.code_execution.base import CodeExecutionSession
    from private_gpt.components.llm.llm_component import LLMComponent
    from private_gpt.events.models import ResultContentBlockType
    from private_gpt.settings.settings import LLMModelConfig

logger = logging.getLogger(__name__)

READABLE_ROOTS = tuple(
    os.path.normpath(mount.target) for mount in DEFAULT_SESSION_LAYOUT
)
WORKSPACE_TARGET = next(
    mount.target for mount in DEFAULT_SESSION_LAYOUT if mount.access == "rw"
)


def source_path_or_error(filepath: str, tool_name: str) -> str:
    """Return a normalized sandbox path, or raise a clear error for anything else."""
    raw = filepath.strip() if filepath else ""
    candidate = os.path.normpath(raw) if raw else ""
    if candidate and os.path.isabs(candidate):
        for root in READABLE_ROOTS:
            prefix = root.rstrip("/") + "/"
            if candidate.startswith(prefix) and candidate != root:
                return candidate

    accepted = ", ".join(f"{root}/" for root in READABLE_ROOTS)
    raise ValueError(
        f"{tool_name} needs an absolute sandbox path under one of: {accepted}. "
        f"Got {filepath!r}."
    )


async def write_markdown(
    session: CodeExecutionSession,
    source: str,
    text: str,
    write_lock: asyncio.Lock,
) -> str:
    """Write *text* beside the workspace as ``<stem>.md`` and return the target.

    Only the name claim and the write are serialized behind *write_lock* — two
    sources sharing a stem would otherwise both probe the workspace before
    either wrote, and the second would overwrite the first instead of landing on
    ``(2).md``.
    """
    desired = f"{PurePosixPath(source).stem}.md"
    async with write_lock:
        target = await unique_name(f"{WORKSPACE_TARGET}{desired}", session.path_exists)
        await session.write_file(target, text.encode())
    return target


async def run_over_paths(
    filepaths: list[str],
    worker: Callable[[str, asyncio.Lock], Awaitable[str]],
    max_concurrency: int | None,
    gerund: str,
    past: str,
    noun: str,
) -> list[ResultContentBlockType]:
    """Run *worker* over every path, reporting each outcome as its own block.

    One slow or broken file must not hold up (or fail) the others, so failures
    become an error block for that path alone.
    """
    semaphore = (
        asyncio.Semaphore(max_concurrency)
        if max_concurrency and max_concurrency > 0
        else None
    )
    write_lock = asyncio.Lock()

    async def _bounded(filepath: str) -> str:
        if semaphore is not None:
            async with semaphore:
                return await worker(filepath, write_lock)
        return await worker(filepath, write_lock)

    results = await asyncio.gather(
        *[_bounded(filepath) for filepath in filepaths],
        return_exceptions=True,
    )

    blocks: list[ResultContentBlockType] = []
    succeeded = 0
    for filepath, result in zip(filepaths, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning("Could not %s %s", gerund, filepath, exc_info=result)
            blocks.append(TextBlock(text=f"Error {gerund} {filepath}: {result}"))
        else:
            blocks.append(TextBlock(text=result))
            succeeded += 1

    blocks.append(
        TextBlock(
            text=(
                f"{past.capitalize()} {succeeded} of {len(filepaths)} {noun}(s)."
                if filepaths
                else f"No {noun}s were requested."
            )
        )
    )
    return blocks


def resolve_media_llm(
    llm_component: LLMComponent,
    supports: Callable[[LLM, LLMModelConfig], bool],
    modality: str,
) -> LLM:
    """Return a model that can handle *modality*, preferring the default one.

    The tool is built without a ``ChatState``, so it cannot inherit the
    request's model the way ``MultimodalRequestInterceptor`` does. It resolves
    at call time instead, which also means a tool rebuilt under Celery picks the
    same model as one built in-process.
    """
    try:
        default_llm = llm_component.get_llm()
        if supports(default_llm, llm_component.get_config()):
            return default_llm
    except ValueError:
        logger.debug("No default model configured; falling back to a capable one")

    found = next(llm_component.filter(supports), None)
    if found is None:
        raise ValueError(f"No {modality}-capable model is configured.")
    return found[0]
