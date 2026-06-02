import asyncio
import builtins
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.base.llms.types import AudioBlock
from llama_index.core.llms import LLM, ChatMessage
from pydantic import Field

from private_gpt.components.llm.custom.mock import FunctionCallingLLMMock
from private_gpt.components.multimodality.audio_handler import (
    AudioProcessingWorkflow,
    TimestampSegment,
    TranscriptionContent,
    TranscriptionEvaluation,
    TranscriptionStrategy,
)


class MockFlexibleModel:
    def __init__(self, **kwargs: Any) -> None:
        self._data = kwargs

    def dict(self) -> dict[str, Any]:
        return self._data

    def model_dump(self) -> builtins.dict[str, Any]:
        return self._data

    def __getattr__(self, item: str) -> Any:
        return self._data.get(item)


class MockLLM(FunctionCallingLLMMock):
    responses: list[Any] = Field(default_factory=list)
    call_count: int = Field(default=0)
    messages_history: list[list[ChatMessage]] = Field(default_factory=list)
    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)

    def __init__(self, responses: list[Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if responses is not None:
            self.responses = responses
        self.responses = self.responses or []
        self.call_count = 0
        self.messages_history = []
        self.lock = asyncio.Lock()

    async def astructured_chat(
        self, output_cls: type, messages: list[ChatMessage], **kwargs: Any
    ) -> Any:
        async with self.lock:
            self.messages_history.append(messages)
            if self.call_count < len(self.responses):
                response = self.responses[self.call_count]
                self.call_count += 1
                return response
        raise IndexError("No more mock responses available")


class PatchAudioBlock(AudioBlock):
    """Patch AudioBlock to avoid validation issues in tests."""

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        return f"PatchAudioBlock(url={self.url})"

    def __str__(self) -> str:
        return self.__repr__()


def create_test_audio_block() -> AudioBlock:
    return PatchAudioBlock(
        url="https://commondatastorage.googleapis.com/codeskulptor-demos/DDR_assets/Kangaroo_MusiQue_-_The_Neverwritten_Role_Playing_Game.mp3"
    )


test_audio_blocks: list[AudioBlock] = [create_test_audio_block()]


@pytest.mark.asyncio
async def test_clean_forward_flow() -> None:
    """Test 1: Clean forward flow without preprocessing or evaluation."""
    mock_responses = [
        TranscriptionStrategy(
            type="speech",
            confidence=0.9,
            language="en",
            has_multiple_speakers=False,
            has_background_noise=False,
            enhance_audio=False,
            speaker_diarization=False,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Hello, how are you today?",
                )
            ],
            is_complete=True,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 2",
                )
            ],
            is_complete=True,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 3",
                )
            ],
            is_complete=True,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 4",
                )
            ],
            is_complete=True,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 5",
                )
            ],
            is_complete=True,
        ),
    ]

    mock_llm = MockLLM(responses=mock_responses)
    workflow = AudioProcessingWorkflow(
        audio_multimodal_llm=mock_llm,
        num_max_retries=1,
    )
    result = await workflow.run(
        audio_blocks=test_audio_blocks,
        user_query="Transcribe this audio",
        max_iterations=1,
        enable_preprocessing=False,
        enable_evaluation=False,
        kwargs={},
    )

    assert result is not None
    assert "Hello" in result.transcript
    assert result.evaluation is None
    assert mock_llm.call_count == 6


@pytest.mark.asyncio
async def test_incomplete_content_with_retries() -> None:
    """Test 2: Evaluation=False, each chunk generates incomplete content."""
    mock_responses = [
        TranscriptionStrategy(
            type="conversation",
            confidence=0.8,
            language="en",
            has_multiple_speakers=True,
            has_background_noise=True,
            enhance_audio=True,
            speaker_diarization=True,
        ),
        # Chunk 1: iteration 1 (incomplete) + iteration 2 (complete)
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Partial transcription...",
                )
            ],
            is_complete=False,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=5.0,
                    speaker="Speaker 2",
                    text="This is the complete transcription.",
                )
            ],
            is_complete=True,
        ),
        # Chunk 2: iteration 1 (incomplete) + iteration 2 (complete)
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 2 partial",
                )
            ],
            is_complete=False,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 2 complete",
                )
            ],
            is_complete=True,
        ),
        # Chunk 3: iteration 1 (incomplete) + iteration 2 (complete)
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 3 partial",
                )
            ],
            is_complete=False,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 3 complete",
                )
            ],
            is_complete=True,
        ),
        # Chunk 4: iteration 1 (incomplete) + iteration 2 (complete)
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 4 partial",
                )
            ],
            is_complete=False,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 4 complete",
                )
            ],
            is_complete=True,
        ),
        # Chunk 5: iteration 1 (incomplete) + iteration 2 (complete)
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 5 partial",
                )
            ],
            is_complete=False,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 5 complete",
                )
            ],
            is_complete=True,
        ),
    ]

    mock_llm = MockLLM(responses=mock_responses)
    workflow = AudioProcessingWorkflow(audio_multimodal_llm=mock_llm)

    result = await workflow.run(
        audio_blocks=test_audio_blocks,
        user_query="Transcribe conversation",
        max_iterations=2,
        enable_preprocessing=True,
        enable_evaluation=False,
        kwargs={},
    )

    assert result is not None
    assert "Partial transcription" in result.transcript
    assert "complete transcription" in result.transcript
    assert result.evaluation is None
    assert mock_llm.call_count == 11


@pytest.mark.asyncio
async def test_evaluation_passes() -> None:
    """Test 3: Evaluation=True, content is complete and passes evaluation."""
    mock_responses = [
        TranscriptionStrategy(
            type="lecture",
            confidence=0.95,
            language="en",
            has_multiple_speakers=False,
            has_background_noise=False,
            enhance_audio=False,
            speaker_diarization=False,
        ),
        # Chunk 1: transcription + evaluation
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=10.0,
                    speaker="Speaker 1",
                    text="Welcome to this lecture on machine learning.",
                )
            ],
            is_complete=True,
        ),
        TranscriptionEvaluation(
            score=0.9,
            issues_found=[],
            clarity=0.95,
        ),
        # Chunk 2: transcription + evaluation
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 2",
                )
            ],
            is_complete=True,
        ),
        TranscriptionEvaluation(
            score=0.9,
            issues_found=[],
            clarity=0.95,
        ),
        # Chunk 3: transcription + evaluation
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 3",
                )
            ],
            is_complete=True,
        ),
        TranscriptionEvaluation(
            score=0.9,
            issues_found=[],
            clarity=0.95,
        ),
        # Chunk 4: transcription + evaluation
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 4",
                )
            ],
            is_complete=True,
        ),
        TranscriptionEvaluation(
            score=0.9,
            issues_found=[],
            clarity=0.95,
        ),
        # Chunk 5: transcription + evaluation
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 5",
                )
            ],
            is_complete=True,
        ),
        TranscriptionEvaluation(
            score=0.9,
            issues_found=[],
            clarity=0.95,
        ),
    ]

    mock_llm = MockLLM(responses=mock_responses)
    workflow = AudioProcessingWorkflow(
        audio_multimodal_llm=mock_llm,
        num_max_retries=1,
        max_workers=1,
    )

    result = await workflow.run(
        audio_blocks=test_audio_blocks,
        user_query="Transcribe lecture",
        max_iterations=2,
        enable_preprocessing=True,
        enable_evaluation=True,
        kwargs={},
    )

    assert result is not None
    assert "machine learning" in result.transcript
    assert result.evaluation is None  # Parent doesn't return evaluation
    assert mock_llm.call_count == 11


@pytest.mark.asyncio
async def test_evaluation_fails_triggers_retry() -> None:
    """Test 4: Evaluation fails, triggers retry with errors/suggestions."""
    mock_responses = [
        TranscriptionStrategy(
            type="interview",
            confidence=0.7,
            language="en",
            has_multiple_speakers=True,
            has_background_noise=True,
            enhance_audio=True,
            speaker_diarization=True,
        ),
        # Chunk 1: transcription + evaluation (fail)
        # + retry transcription + evaluation (pass)
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0, end=2.5, speaker="Speaker 1", text="Unclear audio"
                )
            ],
            is_complete=True,
        ),
        TranscriptionEvaluation(
            score=0.5,
            issues_found=["Background noise interference", "Speaker overlap unclear"],
            clarity=0.4,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=3.0,
                    speaker="Speaker 1",
                    text="Hello, can you tell us about your project?",
                ),
                TimestampSegment(
                    start=3.5,
                    end=6.0,
                    speaker="Speaker 2",
                    text="Yes, we are working on AI solutions.",
                ),
            ],
            is_complete=True,
        ),
        TranscriptionEvaluation(
            score=0.85,
            issues_found=[],
            clarity=0.8,
        ),
        # Chunk 2: transcription + evaluation (fail)
        # + retry transcription + evaluation (pass)
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 2 unclear",
                )
            ],
            is_complete=True,
        ),
        TranscriptionEvaluation(
            score=0.5,
            issues_found=["Background noise"],
            clarity=0.4,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 2 clear",
                )
            ],
            is_complete=True,
        ),
        TranscriptionEvaluation(
            score=0.85,
            issues_found=[],
            clarity=0.8,
        ),
        # Chunk 3: transcription + evaluation (fail)
        # + retry transcription + evaluation (pass)
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 3 unclear",
                )
            ],
            is_complete=True,
        ),
        TranscriptionEvaluation(
            score=0.5,
            issues_found=["Background noise"],
            clarity=0.4,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 3 clear",
                )
            ],
            is_complete=True,
        ),
        TranscriptionEvaluation(
            score=0.85,
            issues_found=[],
            clarity=0.8,
        ),
        # Chunk 4: transcription + evaluation (fail)
        # + retry transcription + evaluation (pass)
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 4 unclear",
                )
            ],
            is_complete=True,
        ),
        TranscriptionEvaluation(
            score=0.5,
            issues_found=["Background noise"],
            clarity=0.4,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 4 clear",
                )
            ],
            is_complete=True,
        ),
        TranscriptionEvaluation(
            score=0.85,
            issues_found=[],
            clarity=0.8,
        ),
        # Chunk 5: transcription + evaluation (fail)
        # + retry transcription + evaluation (pass)
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 5 unclear",
                )
            ],
            is_complete=True,
        ),
        TranscriptionEvaluation(
            score=0.5,
            issues_found=["Background noise"],
            clarity=0.4,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 5 clear",
                )
            ],
            is_complete=True,
        ),
        TranscriptionEvaluation(
            score=0.85,
            issues_found=[],
            clarity=0.8,
        ),
    ]

    mock_llm = MockLLM(responses=mock_responses)
    workflow = AudioProcessingWorkflow(
        audio_multimodal_llm=mock_llm,
        num_max_retries=1,
        max_workers=1,
    )

    result = await workflow.run(
        audio_blocks=test_audio_blocks,
        user_query="Transcribe interview",
        max_iterations=3,
        enable_preprocessing=True,
        enable_evaluation=True,
        kwargs={},
    )

    assert result is not None
    assert "Unclear audio" not in result.transcript
    assert "project" in result.transcript
    assert result.evaluation is None
    assert mock_llm.call_count == 21


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enable_preprocessing", "enable_evaluation", "expected_calls"),
    [
        (False, False, 6),  # 1 strategy + 5 chunks
        (True, False, 6),  # 1 strategy + 5 chunks
        (False, True, 11),  # 1 strategy + 5 chunks * 2 (transcription + evaluation)
        (True, True, 11),  # 1 strategy + 5 chunks * 2 (transcription + evaluation)
    ],
)
async def test_configuration_options(
    enable_preprocessing: bool,
    enable_evaluation: bool,
    expected_calls: int,
) -> None:
    """Test different configuration combinations."""
    mock_responses = [
        TranscriptionStrategy(
            type="speech",
            confidence=0.8,
            language="en",
            has_multiple_speakers=False,
            has_background_noise=False,
            enhance_audio=False,
            speaker_diarization=False,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(start=0.0, end=2.5, speaker="Speaker 1", text="Test")
            ],
            is_complete=True,
        ),
        TranscriptionEvaluation(score=0.9, issues_found=[], clarity=0.85)
        if enable_evaluation
        else None,
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 2",
                )
            ],
            is_complete=True,
        ),
        TranscriptionEvaluation(score=0.9, issues_found=[], clarity=0.85)
        if enable_evaluation
        else None,
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 3",
                )
            ],
            is_complete=True,
        ),
        TranscriptionEvaluation(score=0.9, issues_found=[], clarity=0.85)
        if enable_evaluation
        else None,
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 4",
                )
            ],
            is_complete=True,
        ),
        TranscriptionEvaluation(score=0.9, issues_found=[], clarity=0.85)
        if enable_evaluation
        else None,
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 5",
                )
            ],
            is_complete=True,
        ),
        TranscriptionEvaluation(score=0.9, issues_found=[], clarity=0.85)
        if enable_evaluation
        else None,
    ]

    mock_llm = MockLLM(responses=[resp for resp in mock_responses if resp is not None])
    workflow = AudioProcessingWorkflow(
        audio_multimodal_llm=mock_llm,
        num_max_retries=1,
        max_workers=1,
    )

    result = await workflow.run(
        audio_blocks=test_audio_blocks,
        user_query="Test query",
        max_iterations=2,
        enable_preprocessing=enable_preprocessing,
        enable_evaluation=enable_evaluation,
        kwargs={},
    )

    assert result is not None
    assert mock_llm.call_count == expected_calls


@pytest.mark.asyncio
async def test_max_iterations_reached() -> None:
    """Test 5: Max iterations reached without complete content."""
    mock_responses = [
        TranscriptionStrategy(
            type="speech",
            confidence=0.8,
            language="en",
            has_multiple_speakers=False,
            has_background_noise=False,
            enhance_audio=False,
            speaker_diarization=False,
        ),
        # Chunk 1: iteration 1 (incomplete) + iteration 2 (incomplete, max reached)
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0, end=2.5, speaker="Speaker 1", text="Partial 1"
                )
            ],
            is_complete=False,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0, end=2.5, speaker="Speaker 1", text="Partial 2"
                )
            ],
            is_complete=False,
        ),
        # Chunk 2: iteration 1 (incomplete) + iteration 2 (incomplete, max reached)
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 2 iter 1",
                )
            ],
            is_complete=False,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 2 iter 2",
                )
            ],
            is_complete=False,
        ),
        # Chunk 3: iteration 1 (incomplete) + iteration 2 (incomplete, max reached)
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 3 iter 1",
                )
            ],
            is_complete=False,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 3 iter 2",
                )
            ],
            is_complete=False,
        ),
        # Chunk 4: iteration 1 (incomplete) + iteration 2 (incomplete, max reached)
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 4 iter 1",
                )
            ],
            is_complete=False,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 4 iter 2",
                )
            ],
            is_complete=False,
        ),
        # Chunk 5: iteration 1 (incomplete) + iteration 2 (incomplete, max reached)
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 5 iter 1",
                )
            ],
            is_complete=False,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Chunk 5 iter 2",
                )
            ],
            is_complete=False,
        ),
    ]

    mock_llm = MockLLM(responses=mock_responses)
    workflow = AudioProcessingWorkflow(
        audio_multimodal_llm=mock_llm,
        num_max_retries=1,
        max_workers=1,
    )

    result = await workflow.run(
        audio_blocks=test_audio_blocks,
        user_query="Transcribe audio",
        max_iterations=2,
        enable_preprocessing=False,
        enable_evaluation=False,
        kwargs={},
    )

    assert result is not None
    assert "Partial 1" in result.transcript
    assert "Partial 2" in result.transcript
    assert mock_llm.call_count == 11


@pytest.mark.asyncio
async def test_error_handling_in_strategy_inference() -> None:
    """Test 6: Error handling during strategy inference."""
    mock_llm = MagicMock(spec=LLM)
    mock_llm.astructured_chat = AsyncMock(
        side_effect=RuntimeError("Strategy inference failed")
    )

    workflow = AudioProcessingWorkflow(
        audio_multimodal_llm=mock_llm,
        num_max_retries=1,
        max_workers=1,
    )

    with pytest.raises(RuntimeError):
        await workflow.run(
            audio_blocks=test_audio_blocks,
            user_query="Test error handling",
            max_iterations=1,
            enable_preprocessing=False,
            enable_evaluation=False,
            kwargs={},
        )
