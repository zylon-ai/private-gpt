"""Test download orchestration via HuggingFace Hub."""

from pathlib import Path

import pytest

from private_gpt.components.llm.tokenizers.models.model_downloader import (
    download_from_hf,
    download_model,
)


class TestNetworkFailures:
    """Network and dependency error handling."""

    @pytest.mark.asyncio
    async def test_hf_download_failure_returns_none(
        self,
        hf_cache_dir: Path,
        mock_hf_hub_failure,
    ):
        result = await download_model(
            model_id="test/model",
            cache_dir=hf_cache_dir,
            tokenizer_only=False,
        )

        assert result is None

    def test_hf_hub_not_available(
        self,
        hf_cache_dir: Path,
        mock_hf_hub_not_available,
    ):
        """If huggingface_hub is not installed, return None gracefully."""
        result = download_from_hf(
            model_id="test/model",
            cache_dir=hf_cache_dir,
            tokenizer_only=False,
        )

        assert result is None


class TestTokenizerOnlyDownload:
    """Tokenizer-only vs full-model download behavior."""

    @pytest.mark.asyncio
    async def test_tokenizer_only_flag(
        self,
        hf_cache_dir: Path,
        mock_hf_hub_success,
    ):
        result = await download_model(
            model_id="test/model",
            cache_dir=hf_cache_dir,
            tokenizer_only=True,
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_full_model_download(
        self,
        hf_cache_dir: Path,
        mock_hf_hub_success,
    ):
        result = await download_model(
            model_id="test/model",
            cache_dir=hf_cache_dir,
            tokenizer_only=False,
        )

        assert result is not None


class TestDownloadValidation:
    """Downloaded artifacts must pass local validation."""

    @pytest.mark.asyncio
    async def test_invalid_download_result_returns_none(
        self,
        hf_cache_dir: Path,
        monkeypatch,
    ):
        def mock_snapshot_download(*args, **kwargs) -> str:
            invalid_dir = hf_cache_dir / "invalid_download"
            invalid_dir.mkdir(parents=True, exist_ok=True)
            return str(invalid_dir)

        import sys

        mock_hub = type("MockHub", (), {})()
        mock_hub.snapshot_download = mock_snapshot_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", mock_hub)

        result = await download_model(
            model_id="test/model",
            cache_dir=hf_cache_dir,
            tokenizer_only=False,
        )

        assert result is None
