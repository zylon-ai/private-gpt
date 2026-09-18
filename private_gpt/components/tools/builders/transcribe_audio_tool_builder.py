from __future__ import annotations

import logging
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, cast

from injector import inject, singleton
from llama_index.core.base.llms.types import AudioBlock as LIAudioBlock

from private_gpt.components.chat.models.chat_config_models import (
    ToolRequirements,
    ToolSpec,
)
from private_gpt.components.code_execution.code_execution_component import (
    CodeExecutionComponent,
)
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.llm.llm_helper import supports_audio
from private_gpt.components.multimodality.audio_handler import transcribe_audio
from private_gpt.components.tools.builders.sandbox_file_tools import (
    resolve_media_llm,
    run_over_paths,
    source_path_or_error,
    write_markdown,
)
from private_gpt.components.tools.events.adapters import TranscribeAudioEventAdapter
from private_gpt.components.tools.remote_execution import build_rebuild_metadata
from private_gpt.components.tools.tool_names import TRANSCRIBE_AUDIO_TOOL_NAME
from private_gpt.components.tools.tool_placeholders import TRANSCRIBE_AUDIO_TOOL_FN
from private_gpt.di import get_global_injector
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    import asyncio

    from llama_index.core.llms import LLM

    from private_gpt.components.code_execution.base import (
        CodeExecutionSession,
        CodeExecutionSessionConfig,
    )
    from private_gpt.events.models import ResultContentBlockType

logger = logging.getLogger(__name__)


async def _transcribe_one(
    filepath: str,
    session: CodeExecutionSession,
    audio_llm: LLM,
    write_lock: asyncio.Lock,
) -> str:
    """Transcribe one sandbox audio file into the workspace."""
    source = source_path_or_error(filepath, TRANSCRIBE_AUDIO_TOOL_NAME)
    if not await session.path_exists(source):
        raise FileNotFoundError(f"File not found: {source}")

    raw = await session.read_file(source)
    if not raw:
        raise ValueError(f"{source} is empty.")

    block = LIAudioBlock(
        audio=raw, format=PurePosixPath(source).suffix.lstrip(".") or None
    )
    result = await transcribe_audio(audio_llm, [block])
    transcript = result.transcript if result else None
    if not transcript:
        raise ValueError(f"No speech could be transcribed from {source}.")

    target = await write_markdown(session, source, transcript, write_lock)
    return (
        f"Transcribed {source}. The transcript was left at {target} "
        f"({len(transcript)} characters)."
    )


@singleton
class TranscribeAudioToolBuilder:
    @inject
    def __init__(
        self,
        code_execution_component: CodeExecutionComponent,
        llm_component: LLMComponent,
        settings: Settings,
    ) -> None:
        self._component = code_execution_component
        self._llm_component = llm_component
        self._max_concurrency = settings.chat.preprocess.multimodal.max_concurrency

    async def build_tool(
        self,
        config: CodeExecutionSessionConfig,
        name: str = TRANSCRIBE_AUDIO_TOOL_NAME,
        type: str = TRANSCRIBE_AUDIO_TOOL_NAME + "_v1",
        description: str = TRANSCRIBE_AUDIO_TOOL_FN.metadata.description,
    ) -> ToolSpec:
        async def transcribe_audios(
            filepaths: list[str],
        ) -> list[ResultContentBlockType]:
            session = await self._component.get_or_create_session(config)
            if session is None:
                raise ValueError("code_execution provider is not configured.")

            audio_llm = resolve_media_llm(self._llm_component, supports_audio, "audio")

            async def _worker(filepath: str, write_lock: asyncio.Lock) -> str:
                return await _transcribe_one(filepath, session, audio_llm, write_lock)

            return await run_over_paths(
                filepaths,
                _worker,
                self._max_concurrency,
                gerund="transcribing",
                past="transcribed",
                noun="audio file",
            )

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime="server",
            event_adapter=TranscribeAudioEventAdapter,
            description=description,
            async_fn=transcribe_audios,
            requirements=[ToolRequirements.SANDBOX],
            execution_metadata=build_rebuild_metadata(
                rebuild_transcribe_audio_tool,
                {
                    "config": config,
                    "name": name,
                    "type": type,
                    "description": description,
                },
            ),
        )


async def rebuild_transcribe_audio_tool(**kwargs: Any) -> ToolSpec:
    builder = get_global_injector().get(TranscribeAudioToolBuilder)
    return await builder.build_tool(**cast(Any, kwargs))
