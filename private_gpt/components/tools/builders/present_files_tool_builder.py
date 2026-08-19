from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import (
    ToolRequirements,
    ToolSpec,
)
from private_gpt.components.code_execution.code_execution_component import (
    CodeExecutionComponent,
)
from private_gpt.components.environment.layout import (
    DEFAULT_SESSION_LAYOUT,
    canonical_to_storage_path,
)
from private_gpt.components.tools.events.adapters import PresentFilesEventAdapter
from private_gpt.components.tools.remote_execution import build_rebuild_metadata
from private_gpt.components.tools.tool_names import PRESENT_FILES_TOOL_NAME
from private_gpt.components.tools.tool_placeholders import PRESENT_FILES_TOOL_FN
from private_gpt.di import get_global_injector
from private_gpt.events.models import LocalResourceBlock, TextBlock

if TYPE_CHECKING:
    from private_gpt.components.code_execution.base import (
        CodeExecutionSession,
        CodeExecutionSessionConfig,
    )
    from private_gpt.events.models import ResultContentBlockType

_OUTPUTS_TARGET = next(
    mount.target for mount in DEFAULT_SESSION_LAYOUT if mount.name == "outputs"
)


_EXTENSION_MIME_FALLBACKS: dict[str, str] = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".toml": "application/toml",
    ".jsonl": "application/jsonl",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".sh": "text/x-sh",
    ".py": "text/x-python",
    ".rs": "text/x-rust",
    ".go": "text/x-go",
}


def _encode_file_id(path: str) -> str:
    return base64.urlsafe_b64encode(path.encode()).decode().rstrip("=")


def _is_files_api_path(path: str) -> bool:
    """Return True when Files API can resolve *path* via the session layout."""
    return canonical_to_storage_path(path) != path


def _outputs_copy_path(path: str) -> str:
    posix = Path(path)
    name = posix.name or "file"
    parent = posix.parent.name
    if parent and parent not in {".", "/", "outputs"}:
        return f"{_OUTPUTS_TARGET}{parent}/{name}"
    return f"{_OUTPUTS_TARGET}{name}"


async def _unique_outputs_path(session: CodeExecutionSession, dest: str) -> str:
    if not await session.path_exists(dest):
        return dest
    posix = Path(dest)
    stem = posix.stem or "file"
    suffix = posix.suffix
    parent = str(posix.parent)
    for n in range(2, 1000):
        candidate = f"{parent}/{stem}-{n}{suffix}"
        if not await session.path_exists(candidate):
            return candidate
    raise FileExistsError(f"Could not find a unique outputs path for {dest}")


async def _materialize_for_files_api(
    session: CodeExecutionSession, filepath: str
) -> str:
    """Copy sandbox-only files into outputs so GET /v1/files can serve them.

    present_files can see any path that exists in the live sandbox, but the
    Files API only stores session mounts (workspace, uploads, outputs). The
    backend promote-to-artifact interceptor downloads via that API, so files
    under e.g. ``/home/agent/.claude/skills/`` 404 unless copied first.
    """
    if _is_files_api_path(filepath):
        return filepath
    content = await session.read_file(filepath)
    dest = await _unique_outputs_path(session, _outputs_copy_path(filepath))
    await session.write_file(dest, content)
    return dest


@singleton
class PresentFilesToolBuilder:
    @inject
    def __init__(self, code_execution_component: CodeExecutionComponent) -> None:
        self._component = code_execution_component

    async def build_tool(
        self,
        config: CodeExecutionSessionConfig,
        name: str = PRESENT_FILES_TOOL_NAME,
        type: str = PRESENT_FILES_TOOL_NAME + "_v1",
        description: str = PRESENT_FILES_TOOL_FN.metadata.description,
    ) -> ToolSpec:
        async def present_files(filepaths: list[str]) -> list[ResultContentBlockType]:
            session = await self._component.get_or_create_session(config)
            if session is None:
                raise ValueError("code_execution provider is not configured.")

            blocks: list[ResultContentBlockType] = []
            presented: list[str] = []
            for filepath in filepaths:
                try:
                    if not await session.path_exists(filepath):
                        raise FileNotFoundError(f"File not found: {filepath}")
                    presented_path = await _materialize_for_files_api(session, filepath)
                    mime_type, _ = mimetypes.guess_type(presented_path)
                    if mime_type is None:
                        suffix = Path(presented_path).suffix.lower()
                        mime_type = _EXTENSION_MIME_FALLBACKS.get(
                            suffix, "application/octet-stream"
                        )
                    blocks.append(
                        LocalResourceBlock(
                            file_path=presented_path,
                            file_id=_encode_file_id(presented_path),
                            name=Path(presented_path).stem,
                            mime_type=mime_type,
                        )
                    )
                    presented.append(Path(presented_path).name)
                except Exception as exc:
                    blocks.append(TextBlock(text=f"Error presenting {filepath}: {exc}"))
            blocks.append(
                TextBlock(
                    text=f"Presented {len(presented)} file(s): {', '.join(presented)}"
                    if presented
                    else "No files could be presented."
                )
            )
            return blocks

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime="server",
            event_adapter=PresentFilesEventAdapter,
            description=description,
            async_fn=present_files,
            requirements=[ToolRequirements.SANDBOX],
            execution_metadata=build_rebuild_metadata(
                rebuild_present_files_tool,
                {
                    "config": config,
                    "name": name,
                    "type": type,
                    "description": description,
                },
            ),
        )


async def rebuild_present_files_tool(**kwargs: Any) -> ToolSpec:
    builder = get_global_injector().get(PresentFilesToolBuilder)
    return await builder.build_tool(**cast(Any, kwargs))
