import logging
from typing import Any, Literal

from llama_index.core.base.llms.types import AudioBlock, TextBlock
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import LLM, ChatMessage, MessageRole
from llama_index.core.program.utils import FlexibleModel
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from pydantic import BaseModel, Field
from workflows.errors import WorkflowRuntimeError
from workflows.resource import ResourceManager

from private_gpt.artifact_index.artifact_exception import ModelNotAvailableError
from private_gpt.components.llm.models import MODEL_NOT_AVAILABLE_EXCEPTION_TYPES
from private_gpt.components.prompts.prompt_builder import PromptBuilderService
from private_gpt.components.workflows.types import AnyContext
from private_gpt.di import get_global_injector
from private_gpt.utils.dependencies import format_missing_dependency_message
from private_gpt.utils.retry import retry_context

logger = logging.getLogger(__name__)
logger.setLevel(logging.FATAL)

_DEFAULT_NUM_WORKERS = 4
_DEFAULT_RETRY_NUMBER = 3
_JITTER = (15.0, 30.0)

AUDIO_NOT_PROCESSABLE = "[Audio not processable]"

_DEFAULT_MAX_AUDIO_DURATION_SECONDS = 28.0
_DEFAULT_CHUNK_OVERLAP_SECONDS = 2.0


class AudioChunk(BaseModel):
    audio_block: AudioBlock = Field(description="Audio block for this chunk")
    start_offset: float = Field(description="Start time offset in seconds")
    end_offset: float = Field(description="End time offset in seconds")
    chunk_index: int = Field(description="Index of this chunk in the sequence")
    total_chunks: int = Field(description="Total number of chunks")

    def __repr__(self) -> str:
        return (
            f"AudioChunk(chunk_index={self.chunk_index}, "
            f"total_chunks={self.total_chunks}, "
            f"start_offset={self.start_offset:.2f}s, "
            f"end_offset={self.end_offset:.2f}s)"
        )

    def __str__(self) -> str:
        return self.__repr__()


class TimestampSegment(BaseModel):
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    speaker: str | None = Field(default=None, description="Speaker identifier")
    text: str | None = Field(description="Transcribed text for this segment")

    def __str__(self) -> str:
        return f"[{self.start:.2f}-{self.end:.2f}] ({self.speaker}): {self.text}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimestampSegment":
        return cls(
            start=data.get("start", 0.0),
            end=data.get("end", 0.0),
            speaker=data.get("speaker"),
            text=data.get("text", ""),
        )


class TranscriptionStrategy(BaseModel):
    type: Literal[
        "speech",
        "music",
        "conversation",
        "lecture",
        "podcast",
        "interview",
        "ambient",
        "mixed",
    ] = Field(description="Primary audio content type detected")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in the detection (0-1)"
    )
    language: str = Field(default="en", description="Detected language code")
    has_multiple_speakers: bool = Field(
        default=False, description="Whether audio contains multiple speakers"
    )
    has_background_noise: bool = Field(
        default=False, description="Whether audio has significant background noise"
    )
    enhance_audio: bool | None = Field(
        default=None, description="Whether to apply audio enhancement"
    )
    speaker_diarization: bool | None = Field(
        default=None, description="Whether to perform speaker diarization"
    )


class TranscriptionContent(BaseModel):
    timestamps: list[TimestampSegment] = Field(
        description="Timestamped segments with speaker and text"
    )
    is_complete: bool | None = Field(
        default=None, description="True if transcription is finished"
    )


class TranscriptionEvaluation(BaseModel):
    score: float = Field(
        ge=0.0, le=1.0, description="Overall transcription quality score"
    )
    issues_found: list[str] = Field(
        default_factory=list, description="Issues identified in transcription"
    )
    clarity: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Audio clarity score"
    )


class ChunkWorkflowInputEvent(StartEvent):
    chunk: AudioChunk
    user_query: str | None
    strategy: TranscriptionStrategy
    enable_speaker_diarization: bool
    max_iterations: int
    enable_evaluation: bool
    kwargs: dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ChunkWorkflowInputEvent(chunk_index={self.chunk.chunk_index}, "
            f"max_iterations={self.max_iterations}, "
            f"enable_evaluation={self.enable_evaluation})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    class Config:
        arbitrary_types_allowed = True


class ChunkTranscriptionResultEvent(Event):
    timestamps: list[TimestampSegment]
    is_complete: bool


class ChunkEvaluationResultEvent(Event):
    timestamps: list[TimestampSegment]
    evaluation: TranscriptionEvaluation | None
    should_retry: bool


class ChunkWorkflowResultEvent(StopEvent):
    chunk_index: int
    timestamps: list[TimestampSegment]
    is_complete: bool


class AudioProcessingInputEvent(StartEvent):
    audio_blocks: list[AudioBlock]
    user_query: str | None = None
    max_iterations: int = 3
    enable_preprocessing: bool = True
    enable_evaluation: bool = True
    enable_speaker_diarization: bool = False
    max_audio_duration: float = _DEFAULT_MAX_AUDIO_DURATION_SECONDS
    chunk_overlap: float = _DEFAULT_CHUNK_OVERLAP_SECONDS
    kwargs: dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"AudioProcessingInputEvent(num_audio_blocks={len(self.audio_blocks)}, "
            f"max_iterations={self.max_iterations}, "
            f"enable_preprocessing={self.enable_preprocessing}, "
            f"enable_evaluation={self.enable_evaluation})"
        )

    def __str__(self) -> str:
        return self.__repr__()


class WorkflowInitializedEvent(Event):
    audio_blocks: list[AudioBlock]

    def __repr__(self) -> str:
        return f"WorkflowInitializedEvent(num_audio_blocks={len(self.audio_blocks)})"

    def __str__(self) -> str:
        return self.__repr__()


class AudioChunkedEvent(Event):
    audio_chunks: list[AudioChunk]


class StrategyInferredEvent(Event):
    strategy: TranscriptionStrategy
    audio_chunks: list[AudioChunk]


class AudioPreprocessedEvent(Event):
    processed_chunks: list[AudioChunk]


class AudioProcessingResultEvent(StopEvent):
    transcript: str
    speakers: list[str] | None = None
    timestamps: list[TimestampSegment] | None = None
    evaluation: TranscriptionEvaluation | None = None


class ChunkTranscriptionWorkflow(Workflow):
    def __init__(
        self,
        audio_multimodal_llm: LLM,
        prompt_builder: PromptBuilderService,
        timeout: float | None = None,
        num_max_retries: int = _DEFAULT_RETRY_NUMBER,
        retry_jitter: tuple[float, float] = _JITTER,
        **kwargs: Any,
    ) -> None:
        super().__init__(timeout=timeout, **kwargs)
        self._llm = audio_multimodal_llm
        self._prompt_builder = prompt_builder
        self._num_max_retries = num_max_retries
        self._retry_jitter = retry_jitter

    @step
    async def init_chunk_workflow(
        self, ctx: AnyContext, ev: ChunkWorkflowInputEvent
    ) -> ChunkTranscriptionResultEvent | ChunkWorkflowResultEvent:
        await ctx.store.set("chunk", ev.chunk)
        await ctx.store.set("user_query", ev.user_query)
        await ctx.store.set("strategy", ev.strategy)
        await ctx.store.set("enable_speaker_diarization", ev.enable_speaker_diarization)
        await ctx.store.set("max_iterations", ev.max_iterations)
        await ctx.store.set("enable_evaluation", ev.enable_evaluation)
        await ctx.store.set("kwargs", ev.kwargs)
        await ctx.store.set("iteration", 1)
        await ctx.store.set("all_timestamps", [])
        await ctx.store.set("continue_content", None)
        await ctx.store.set("issues_found", [])

        return await self._transcribe_and_iterate(ctx)

    async def _transcribe_and_iterate(
        self, ctx: AnyContext
    ) -> ChunkTranscriptionResultEvent | ChunkWorkflowResultEvent:
        chunk = await ctx.store.get("chunk")
        iteration = await ctx.store.get("iteration")
        max_iterations = await ctx.store.get("max_iterations")

        if iteration > max_iterations:
            all_timestamps = await ctx.store.get("all_timestamps", [])
            return ChunkWorkflowResultEvent(
                chunk_index=chunk.chunk_index,
                timestamps=all_timestamps,
                is_complete=True,
            )

        user_query = await ctx.store.get("user_query")
        strategy = await ctx.store.get("strategy")
        enable_speaker_diarization = await ctx.store.get("enable_speaker_diarization")
        continue_content = await ctx.store.get("continue_content")
        issues_found = await ctx.store.get("issues_found")
        kwargs = await ctx.store.get("kwargs")

        transcription_content = await self._transcribe_chunk(
            chunk=chunk,
            user_query=user_query,
            strategy=strategy,
            enable_speaker_diarization=enable_speaker_diarization,
            continue_content=continue_content,
            issues_found=issues_found,
            iteration=iteration,
            **kwargs,
        )

        timestamps = getattr(transcription_content, "timestamps", [])
        is_complete = getattr(transcription_content, "is_complete", False)

        adjusted_timestamps: list[TimestampSegment] = []
        for segment in timestamps:
            adjusted_segment = TimestampSegment(
                start=segment.start + chunk.start_offset,
                end=segment.end + chunk.start_offset,
                speaker=segment.speaker,
                text=segment.text,
            )
            adjusted_timestamps.append(adjusted_segment)

        enable_evaluation = await ctx.store.get("enable_evaluation")

        if enable_evaluation:
            return ChunkTranscriptionResultEvent(
                timestamps=adjusted_timestamps,
                is_complete=is_complete,
            )

        all_timestamps = await ctx.store.get("all_timestamps", [])
        all_timestamps.extend(adjusted_timestamps)
        await ctx.store.set("all_timestamps", all_timestamps)

        if is_complete or iteration >= max_iterations:
            return ChunkWorkflowResultEvent(
                chunk_index=chunk.chunk_index,
                timestamps=all_timestamps,
                is_complete=True,
            )

        await ctx.store.set("iteration", iteration + 1)
        continue_content = " ".join(seg.text for seg in all_timestamps if seg.text)
        await ctx.store.set("continue_content", continue_content)
        await ctx.store.set("issues_found", [])

        return await self._transcribe_and_iterate(ctx)

    @step
    async def evaluate_chunk(
        self, ctx: AnyContext, ev: ChunkTranscriptionResultEvent
    ) -> ChunkEvaluationResultEvent:
        chunk = await ctx.store.get("chunk")
        iteration = await ctx.store.get("iteration")
        max_iterations = await ctx.store.get("max_iterations")
        all_timestamps = await ctx.store.get("all_timestamps", [])

        all_timestamps.extend(ev.timestamps)
        await ctx.store.set("all_timestamps", all_timestamps)

        if iteration > max_iterations:
            return ChunkEvaluationResultEvent(
                timestamps=all_timestamps,
                evaluation=None,
                should_retry=False,
            )

        transcript = " ".join(seg.text for seg in ev.timestamps if seg.text)
        audio_blocks = [chunk.audio_block]

        evaluation = await self._evaluate_transcription(
            transcript=transcript,
            audio_blocks=audio_blocks,
            iteration=iteration,
        )

        should_retry = False
        if evaluation and (evaluation.score < 0.7 or evaluation.issues_found):
            if iteration < max_iterations:
                should_retry = True

        return ChunkEvaluationResultEvent(
            timestamps=all_timestamps,
            evaluation=evaluation,
            should_retry=should_retry,
        )

    @step
    async def decide_next_action(
        self, ctx: AnyContext, ev: ChunkEvaluationResultEvent
    ) -> ChunkTranscriptionResultEvent | ChunkWorkflowResultEvent:
        chunk = await ctx.store.get("chunk")
        iteration = await ctx.store.get("iteration")
        max_iterations = await ctx.store.get("max_iterations")

        if ev.should_retry and iteration < max_iterations:
            await ctx.store.set("iteration", iteration + 1)
            continue_content = " ".join(seg.text for seg in ev.timestamps if seg.text)
            await ctx.store.set("continue_content", continue_content)
            issues_found = ev.evaluation.issues_found if ev.evaluation else []
            await ctx.store.set("issues_found", issues_found)
            await ctx.store.set("all_timestamps", [])
            return await self._transcribe_and_iterate(ctx)

        return ChunkWorkflowResultEvent(
            chunk_index=chunk.chunk_index,
            timestamps=ev.timestamps,
            is_complete=True,
        )

    async def _transcribe_chunk(
        self,
        chunk: AudioChunk,
        user_query: str | None,
        strategy: TranscriptionStrategy,
        enable_speaker_diarization: bool,
        continue_content: str | None,
        issues_found: list[str],
        iteration: int,
        **kwargs: Any,
    ) -> TranscriptionContent | FlexibleModel:
        template = self._prompt_builder.create_audio_transcription_prompt(
            user_query=user_query,
            last_content=continue_content,
            audio_type=strategy.type,
            confidence=strategy.confidence,
            language=strategy.language,
            has_multiple_speakers=strategy.has_multiple_speakers,
            enable_speaker_diarization=enable_speaker_diarization,
            errors=issues_found or None,
        )

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM, blocks=[TextBlock(text=template.format())]
            ),
            ChatMessage(
                role=MessageRole.USER,
                blocks=[
                    TextBlock(
                        text=(
                            f"This is chunk {chunk.chunk_index + 1} of {chunk.total_chunks}. "
                            f"Time range: {chunk.start_offset:.2f}s - {chunk.end_offset:.2f}s."
                            "\nTranscribe the content from this audio:"
                        )
                    ),
                    chunk.audio_block,
                ],
            ),
        ]

        response: TranscriptionContent | FlexibleModel = await self._astructured_chat(
            TranscriptionContent,
            messages,
            seed=iteration + chunk.chunk_index,
            allow_flexible=True,
            **kwargs,
        )

        return response

    async def _evaluate_transcription(
        self,
        transcript: str,
        audio_blocks: list[AudioBlock],
        iteration: int,
    ) -> TranscriptionEvaluation | None:
        if not transcript:
            return None

        eval_template = self._prompt_builder.create_audio_evaluation_prompt()
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM, blocks=[TextBlock(text=eval_template.format())]
            ),
            ChatMessage(
                role=MessageRole.USER,
                blocks=[
                    TextBlock(text=f"Transcript: {transcript}"),
                    *audio_blocks,
                ],
            ),
        ]

        result: TranscriptionEvaluation | FlexibleModel = await self._astructured_chat(
            TranscriptionEvaluation,
            messages,
            seed=iteration,
        )
        if not isinstance(result, TranscriptionEvaluation):
            result = TranscriptionEvaluation(**result.model_dump())

        return result

    async def _astructured_chat(
        self,
        response_model: type[BaseModel],
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> Any:
        try:
            async with retry_context(
                tries=self._num_max_retries,
                jitter=self._retry_jitter,
                logger=logger,
            ) as retry:
                seed = kwargs.pop("seed", None) or 0
                count = 0

                async def _call() -> Any:
                    nonlocal count
                    count += 1

                    structured_chat = getattr(self._llm, "astructured_chat", None)
                    if not callable(structured_chat):
                        raise NotImplementedError(
                            "LLM does not support structured chat."
                        )

                    new_kwargs = kwargs.copy()
                    new_kwargs["seed"] = str(seed) + str(count)

                    return await structured_chat(response_model, messages, **new_kwargs)

                return await retry(_call)
        except MODEL_NOT_AVAILABLE_EXCEPTION_TYPES as e:
            raise ModelNotAvailableError(
                "Model server is not available or request failed."
            ) from e
        except Exception:
            raise


class AudioProcessingWorkflow(Workflow):
    def __init__(
        self,
        audio_multimodal_llm: LLM,
        prompt_builder: PromptBuilderService | None = None,
        callback_manager: CallbackManager | None = None,
        timeout: float | None = 360000.0,
        disable_validation: bool = False,
        verbose: bool = False,
        resource_manager: ResourceManager | None = None,
        num_concurrent_runs: int | None = None,
        max_workers: int = _DEFAULT_NUM_WORKERS,
        num_max_retries: int = _DEFAULT_RETRY_NUMBER,
        retry_jitter: tuple[float, float] = _JITTER,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            timeout=timeout,
            disable_validation=disable_validation,
            verbose=verbose,
            resource_manager=resource_manager,
            num_concurrent_runs=num_concurrent_runs,
        )
        self._llm = audio_multimodal_llm
        self._prompt_builder = prompt_builder or get_global_injector().get(
            PromptBuilderService
        )
        self._kwargs = kwargs

        if callback_manager:
            self._llm.callback_manager = callback_manager

        self._max_concurrent_chunks = max_workers
        self._num_max_retries = num_max_retries
        self._retry_jitter = retry_jitter

    @step
    async def init_workflow(
        self, ctx: AnyContext, ev: AudioProcessingInputEvent
    ) -> WorkflowInitializedEvent:
        await ctx.store.set("user_query", ev.user_query)
        await ctx.store.set("max_iterations", ev.max_iterations)
        await ctx.store.set("enable_preprocessing", ev.enable_preprocessing)
        await ctx.store.set("enable_evaluation", ev.enable_evaluation)
        await ctx.store.set("enable_speaker_diarization", ev.enable_speaker_diarization)
        await ctx.store.set("max_audio_duration", ev.max_audio_duration)
        await ctx.store.set("chunk_overlap", ev.chunk_overlap)
        await ctx.store.set("kwargs", ev.kwargs)

        return WorkflowInitializedEvent(audio_blocks=ev.audio_blocks)

    @step
    async def chunk_audio(
        self, ctx: AnyContext, ev: WorkflowInitializedEvent
    ) -> AudioChunkedEvent:
        max_duration = await ctx.store.get("max_audio_duration")
        chunk_overlap = await ctx.store.get("chunk_overlap")

        all_chunks: list[AudioChunk] = []
        for audio_block in ev.audio_blocks:
            chunks = await self._split_audio_block(
                audio_block, max_duration, chunk_overlap
            )
            all_chunks.extend(chunks)

        await ctx.store.set("audio_chunks", all_chunks)
        return AudioChunkedEvent(audio_chunks=all_chunks)

    @step
    async def infer_strategy(
        self, ctx: AnyContext, ev: AudioChunkedEvent
    ) -> StrategyInferredEvent | AudioPreprocessedEvent:
        iteration = 1
        enable_preprocessing = await ctx.store.get("enable_preprocessing")
        kwargs = await ctx.store.get("kwargs")

        sample_blocks = [ev.audio_chunks[0].audio_block] if ev.audio_chunks else []
        strategy = await self._infer_strategy(sample_blocks, seed=iteration, **kwargs)
        await ctx.store.set("strategy", strategy)

        return (
            StrategyInferredEvent(strategy=strategy, audio_chunks=ev.audio_chunks)
            if enable_preprocessing
            else AudioPreprocessedEvent(processed_chunks=ev.audio_chunks)
        )

    @step
    async def preprocess_audio(
        self, ctx: AnyContext, ev: StrategyInferredEvent
    ) -> AudioPreprocessedEvent:
        enable_preprocessing = await ctx.store.get("enable_preprocessing")

        processed_chunks = ev.audio_chunks
        if enable_preprocessing and ev.strategy.enhance_audio:
            processed_chunks = [
                AudioChunk(
                    audio_block=await self._preprocess_audio(
                        chunk.audio_block, ev.strategy
                    ),
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks,
                )
                for chunk in ev.audio_chunks
            ]

        await ctx.store.set("strategy", ev.strategy)
        return AudioPreprocessedEvent(processed_chunks=processed_chunks)

    @step
    async def transcribe_all_chunks(
        self, ctx: AnyContext, ev: AudioPreprocessedEvent
    ) -> AudioProcessingResultEvent:
        import asyncio

        strategy = await ctx.store.get("strategy")
        max_iterations = await ctx.store.get("max_iterations")
        user_query = await ctx.store.get("user_query")
        enable_speaker_diarization = await ctx.store.get("enable_speaker_diarization")
        enable_evaluation = await ctx.store.get("enable_evaluation")
        kwargs = await ctx.store.get("kwargs")

        if strategy.type in ["ambient"] and strategy.confidence < 0.3:
            response = (
                "\n\n"
                + self._prompt_builder.create_audio_transcription_response(
                    user_query=user_query,
                    transcript="",
                    speakers=None,
                )
                .format()
                .strip()
                + "\n\n"
            )
            return AudioProcessingResultEvent(
                transcript=response if response.strip() else AUDIO_NOT_PROCESSABLE,
                speakers=None,
                timestamps=None,
                evaluation=None,
            )

        semaphore = asyncio.Semaphore(self._max_concurrent_chunks)

        async def process_chunk_with_semaphore(
            chunk: AudioChunk,
        ) -> ChunkWorkflowResultEvent:
            async with semaphore:
                child_workflow = ChunkTranscriptionWorkflow(
                    audio_multimodal_llm=self._llm,
                    prompt_builder=self._prompt_builder,
                    num_max_retries=self._num_max_retries,
                    retry_jitter=self._retry_jitter,
                )
                return await child_workflow.run(
                    chunk=chunk,
                    user_query=user_query,
                    strategy=strategy,
                    enable_speaker_diarization=enable_speaker_diarization,
                    max_iterations=max_iterations,
                    enable_evaluation=enable_evaluation,
                    kwargs=kwargs,
                )

        tasks = [process_chunk_with_semaphore(chunk) for chunk in ev.processed_chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_timestamps: list[TimestampSegment] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Chunk processing failed: {result}")
                continue
            if isinstance(result, ChunkWorkflowResultEvent):
                all_timestamps.extend(result.timestamps)

        all_timestamps.sort(key=lambda x: x.start)
        merged_timestamps = self._merge_and_group_timestamps([all_timestamps])

        transcript = self._build_transcript_from_timestamps(merged_timestamps)
        speakers = self._extract_speakers_from_timestamps(merged_timestamps)

        response = (
            "\n\n"
            + self._prompt_builder.create_audio_transcription_response(
                user_query=user_query,
                transcript=transcript,
                speakers=speakers,
            )
            .format()
            .strip()
            + "\n\n"
        )

        if not response.strip():
            response = AUDIO_NOT_PROCESSABLE

        return AudioProcessingResultEvent(
            transcript=response,
            speakers=speakers,
            timestamps=merged_timestamps if merged_timestamps else None,
            evaluation=None,
        )

    async def _astructured_chat(
        self,
        response_model: type[BaseModel],
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> Any:
        try:
            async with retry_context(
                tries=self._num_max_retries,
                jitter=self._retry_jitter,
                logger=logger,
            ) as retry:
                seed = kwargs.pop("seed", None) or 0
                count = 0

                async def _call() -> Any:
                    nonlocal count
                    count += 1

                    structured_chat = getattr(self._llm, "astructured_chat", None)
                    if not callable(structured_chat):
                        raise NotImplementedError(
                            "LLM does not support structured chat."
                        )

                    new_kwargs = kwargs.copy()
                    new_kwargs["seed"] = str(seed) + str(count)

                    return await structured_chat(response_model, messages, **new_kwargs)

                return await retry(_call)
        except MODEL_NOT_AVAILABLE_EXCEPTION_TYPES as e:
            raise ModelNotAvailableError(
                "Model server is not available or request failed."
            ) from e

    async def _infer_strategy(
        self, audio_blocks: list[AudioBlock], **kwargs: Any
    ) -> TranscriptionStrategy:
        strategy_prompt = self._prompt_builder.create_audio_strategy_prompt()

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                blocks=[TextBlock(text=strategy_prompt.format())],
            ),
            ChatMessage(
                role=MessageRole.USER,
                blocks=[
                    TextBlock(text="Analyze the following audio:"),
                    *audio_blocks,
                ],
            ),
        ]

        strategy: TranscriptionStrategy | FlexibleModel = await self._astructured_chat(
            TranscriptionStrategy, messages, **kwargs
        )
        if not isinstance(strategy, TranscriptionStrategy):
            strategy = TranscriptionStrategy(**strategy.model_dump())

        return strategy

    async def _split_audio_block(
        self,
        audio_block: AudioBlock,
        max_duration: float,
        overlap: float,
    ) -> list[AudioChunk]:
        try:
            import io

            from pydub import AudioSegment  # ty:ignore[unresolved-import]

            audio_data = audio_block.resolve_audio()
            if isinstance(audio_data, bytes):
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
            else:
                audio_segment = AudioSegment.from_file(audio_data)

            duration_seconds = len(audio_segment) / 1000.0

            if duration_seconds <= max_duration:
                return [
                    AudioChunk(
                        audio_block=audio_block,
                        start_offset=0.0,
                        end_offset=duration_seconds,
                        chunk_index=0,
                        total_chunks=1,
                    )
                ]

            max_duration_ms = int(max_duration * 1000)
            overlap_ms = int(overlap * 1000)
            step_size_ms = max_duration_ms - overlap_ms

            chunks: list[AudioChunk] = []
            start_ms = 0
            chunk_index = 0

            while start_ms < len(audio_segment):
                end_ms = min(start_ms + max_duration_ms, len(audio_segment))
                chunk_segment = audio_segment[start_ms:end_ms]

                output_buffer = io.BytesIO()
                chunk_segment.export(output_buffer, format="wav")
                output_buffer.seek(0)

                chunks.append(
                    AudioChunk(
                        audio_block=AudioBlock(
                            audio=output_buffer.getvalue(), format="audio/wav"
                        ),
                        start_offset=start_ms / 1000.0,
                        end_offset=end_ms / 1000.0,
                        chunk_index=chunk_index,
                        total_chunks=0,
                    )
                )

                chunk_index += 1
                start_ms += step_size_ms

                if end_ms >= len(audio_segment):
                    break

            total_chunks = len(chunks)
            for chunk in chunks:
                chunk.total_chunks = total_chunks

            logger.info(
                f"Split audio ({duration_seconds:.2f}s) into {total_chunks} chunks "
                f"(max {max_duration}s, overlap {overlap}s)"
            )

            return chunks

        except ImportError:
            logger.warning(
                "%s Returning the audio as a single chunk.",
                format_missing_dependency_message(
                    "Audio chunking",
                    extras="media",
                ),
            )
            return [
                AudioChunk(
                    audio_block=audio_block,
                    start_offset=0.0,
                    end_offset=0.0,
                    chunk_index=0,
                    total_chunks=1,
                )
            ]
        except Exception:
            return [
                AudioChunk(
                    audio_block=audio_block,
                    start_offset=0.0,
                    end_offset=0.0,
                    chunk_index=0,
                    total_chunks=1,
                )
            ]

    async def _preprocess_audio(
        self, audio_block: AudioBlock, strategy: TranscriptionStrategy
    ) -> AudioBlock:
        if not strategy.enhance_audio:
            return audio_block

        try:
            import io

            import noisereduce as nr  # ty:ignore[unresolved-import]
            import numpy as np
            from pydub import AudioSegment  # ty:ignore[unresolved-import]
            from pydub.effects import normalize  # ty:ignore[unresolved-import]

            audio_data = audio_block.resolve_audio()
            if isinstance(audio_data, bytes):
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
            else:
                audio_segment = AudioSegment.from_file(audio_data)

            samples = np.array(audio_segment.get_array_of_samples())
            sample_rate = audio_segment.frame_rate

            if strategy.has_background_noise:
                samples = nr.reduce_noise(
                    y=samples.astype(float),
                    sr=sample_rate,
                    stationary=True,
                    prop_decrease=0.8,
                )

            enhanced_segment = AudioSegment(
                samples.tobytes(),
                frame_rate=sample_rate,
                sample_width=audio_segment.sample_width,
                channels=audio_segment.channels,
            )

            enhanced_segment = normalize(enhanced_segment)

            output_buffer = io.BytesIO()
            enhanced_segment.export(output_buffer, format="wav")
            output_buffer.seek(0)

            return AudioBlock(audio=output_buffer.getvalue(), format="audio/wav")

        except ImportError:
            logger.warning(
                format_missing_dependency_message(
                    "Audio preprocessing",
                    extras="media",
                )
            )
            return audio_block
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            return audio_block

    def _merge_and_group_timestamps(
        self, results: list[list[TimestampSegment]]
    ) -> list[TimestampSegment]:
        if not results:
            return []

        all_timestamps: list[TimestampSegment] = []
        for result in results:
            all_timestamps.extend(result)

        if not all_timestamps:
            return []

        all_timestamps.sort(key=lambda x: x.start)

        grouped: list[TimestampSegment] = []
        current_group: TimestampSegment | None = None

        for segment in all_timestamps:
            if (
                current_group is None
                or current_group.speaker != segment.speaker
                or segment.start - current_group.end > 1.0
            ):
                if current_group is not None:
                    grouped.append(current_group)
                current_group = TimestampSegment(
                    start=segment.start,
                    end=segment.end,
                    speaker=segment.speaker,
                    text=segment.text,
                )
            else:
                current_group.end = segment.end
                current_group.text = f"{current_group.text} {segment.text}".strip()

        if current_group is not None:
            grouped.append(current_group)

        return grouped

    def _build_transcript_from_timestamps(
        self, timestamps: list[TimestampSegment]
    ) -> str:
        if not timestamps:
            return ""

        return " ".join(segment.text for segment in timestamps if segment.text)

    def _extract_speakers_from_timestamps(
        self, timestamps: list[TimestampSegment]
    ) -> list[str] | None:
        speakers = [
            segment.speaker for segment in timestamps if segment.speaker is not None
        ]
        unique_speakers = list(dict.fromkeys(speakers))
        return unique_speakers if unique_speakers else None


async def transcribe_audio(
    audio_multimodal_llm: LLM,
    audio_blocks: list[AudioBlock] | None = None,
    user_query: str | None = None,
    max_iterations: int = 1,
    enable_preprocessing: bool = False,
    enable_evaluation: bool = False,
    enable_speaker_diarization: bool = False,
    max_audio_duration: float = _DEFAULT_MAX_AUDIO_DURATION_SECONDS,
    chunk_overlap: float = _DEFAULT_CHUNK_OVERLAP_SECONDS,
    **kwargs: Any,
) -> AudioProcessingResultEvent | None:
    if not audio_blocks:
        return None

    workflow = AudioProcessingWorkflow(audio_multimodal_llm, **kwargs)
    try:
        result: AudioProcessingResultEvent = await workflow.run(
            audio_blocks=audio_blocks,
            user_query=user_query,
            max_iterations=max_iterations,
            enable_preprocessing=enable_preprocessing,
            enable_evaluation=enable_evaluation,
            enable_speaker_diarization=enable_speaker_diarization,
            max_audio_duration=max_audio_duration,
            chunk_overlap=chunk_overlap,
            kwargs=kwargs,
        )
    except WorkflowRuntimeError as e:
        exception: BaseException = e.__cause__ or e
        raise exception  # noqa: B904

    return result


async def process_audio_in_message(
    audio_multimodal_llm: LLM,
    message: ChatMessage,
    user_query: str | None = None,
    enable_preprocessing: bool = False,
    enable_evaluation: bool = False,
    enable_speaker_diarization: bool = False,
    max_audio_duration: float = _DEFAULT_MAX_AUDIO_DURATION_SECONDS,
    chunk_overlap: float = _DEFAULT_CHUNK_OVERLAP_SECONDS,
    **kwargs: Any,
) -> str | None:
    audio_blocks = [block for block in message.blocks if isinstance(block, AudioBlock)]
    if not audio_blocks:
        return message.content or ""

    result = await transcribe_audio(
        audio_multimodal_llm,
        audio_blocks,
        user_query,
        enable_preprocessing=enable_preprocessing,
        enable_evaluation=enable_evaluation,
        enable_speaker_diarization=enable_speaker_diarization,
        max_audio_duration=max_audio_duration,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )
    return result.transcript if result and result.transcript else None
