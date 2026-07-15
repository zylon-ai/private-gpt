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
from private_gpt.components.tools.remote_execution import build_rebuild_metadata
from private_gpt.components.tools.tool_names import PRESENT_FILES_TOOL_NAME
from private_gpt.components.tools.tool_placeholders import PRESENT_FILES_TOOL_FN
from private_gpt.di import get_global_injector
from private_gpt.events.models import LocalResourceBlock, TextBlock

if TYPE_CHECKING:
    from private_gpt.components.sandbox.content_bundle import ContentBundle
    from private_gpt.events.models import ResultContentBlockType


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


@singleton
class PresentFilesToolBuilder:
    @inject
    def __init__(self, code_execution_component: CodeExecutionComponent) -> None:
        self._component = code_execution_component

    async def build_tool(
        self,
        session_id: str,
        bundles: list[ContentBundle] | None = None,
        name: str = PRESENT_FILES_TOOL_NAME,
        type: str = PRESENT_FILES_TOOL_NAME + "_v1",
        description: str = PRESENT_FILES_TOOL_FN.metadata.description,
    ) -> ToolSpec:
        async def present_files(filepaths: list[str]) -> list[ResultContentBlockType]:
            blocks: list[ResultContentBlockType] = []
            presented: list[str] = []
            for filepath in filepaths:
                try:
                    mime_type, _ = mimetypes.guess_type(filepath)
                    if mime_type is None:
                        suffix = Path(filepath).suffix.lower()
                        mime_type = _EXTENSION_MIME_FALLBACKS.get(
                            suffix, "application/octet-stream"
                        )
                    blocks.append(
                        LocalResourceBlock(
                            file_path=filepath,
                            file_id=_encode_file_id(filepath),
                            name=Path(filepath).stem,
                            mime_type=mime_type,
                        )
                    )
                    presented.append(Path(filepath).name)
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
            description=description,
            async_fn=present_files,
            requirements=[ToolRequirements.SANDBOX],
            execution_metadata=build_rebuild_metadata(
                rebuild_present_files_tool,
                {
                    "session_id": session_id,
                    "bundles": bundles,
                    "name": name,
                    "type": type,
                    "description": description,
                },
            ),
        )


async def rebuild_present_files_tool(**kwargs: Any) -> ToolSpec:
    builder = get_global_injector().get(PresentFilesToolBuilder)
    return await builder.build_tool(**cast(Any, kwargs))
