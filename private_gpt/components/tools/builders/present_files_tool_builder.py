from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import magic
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
from private_gpt.events.models import BinaryBlock, TextBlock

if TYPE_CHECKING:
    from private_gpt.components.code_execution.base import CodeExecutionSession
    from private_gpt.components.sandbox.content_bundle import ContentBundle
    from private_gpt.events.models import ResultContentBlockType


@singleton
class PresentFilesToolBuilder:
    @inject
    def __init__(self, code_execution_component: CodeExecutionComponent) -> None:
        self._component = code_execution_component

    async def _session(
        self,
        session_id: str,
        bundles: list[ContentBundle] | None = None,
    ) -> CodeExecutionSession:
        session = await self._component.get_or_create_session(
            session_id, extra_bundles=bundles or None
        )
        if session is None:
            raise ValueError("code_execution provider is not configured.")
        return session

    async def build_tool(
        self,
        session_id: str,
        bundles: list[ContentBundle] | None = None,
        name: str = PRESENT_FILES_TOOL_NAME,
        type: str = PRESENT_FILES_TOOL_NAME + "_v1",
        description: str = PRESENT_FILES_TOOL_FN.metadata.description,
    ) -> ToolSpec:
        session = await self._session(session_id, bundles)

        async def present_files(filepaths: list[str]) -> list[ResultContentBlockType]:
            blocks: list[ResultContentBlockType] = []
            presented: list[str] = []
            for filepath in filepaths:
                try:
                    raw = await session.read_file(filepath)
                    try:
                        mime_type = magic.Magic(mime=True).from_buffer(raw)
                    except Exception:
                        mime_type = "application/octet-stream"
                    blocks.append(
                        BinaryBlock.from_bytes(
                            binary=raw,
                            mime_type=mime_type,
                            filename=Path(filepath).name,
                        )
                    )
                    presented.append(Path(filepath).name)
                except Exception as exc:
                    blocks.append(TextBlock(text=f"Error reading {filepath}: {exc}"))
            blocks.append(
                TextBlock(
                    text=f"Presented {len(presented)} file(s): {', '.join(presented)}"
                    if presented
                    else "No files could be read."
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
