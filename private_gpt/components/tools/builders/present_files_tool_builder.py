from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import (
    ToolRequirements,
    ToolSpec,
)
from private_gpt.components.code_execution.code_execution_component import (
    CodeExecutionComponent,
)
from private_gpt.components.tools.tool_names import PRESENT_FILES_TOOL_NAME
from private_gpt.components.tools.tool_placeholders import PRESENT_FILES_TOOL_FN
from private_gpt.events.models import LocalResourceBlock, TextBlock

if TYPE_CHECKING:
    from private_gpt.components.sandbox.content_bundle import ContentBundle
    from private_gpt.events.models import ResultContentBlockType


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
                        mime_type = "application/octet-stream"
                    blocks.append(
                        LocalResourceBlock(
                            file_path=filepath,
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
        )
