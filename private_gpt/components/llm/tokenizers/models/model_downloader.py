from __future__ import annotations

import logging
from pathlib import Path

from private_gpt.components.llm.tokenizers.models.model_cache import validate_model_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_from_hf(
    model_id: str, cache_dir: Path, tokenizer_only: bool
) -> Path | None:
    """Download a model from HuggingFace Hub via snapshot_download."""
    try:
        from huggingface_hub import (  # ty:ignore[unresolved-import]
            snapshot_download,  # type: ignore[import]
        )

        allow_patterns = (
            ["*.json", "*.model", "vocab.txt", "tokenizer*"] if tokenizer_only else None
        )
        downloaded: str = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
            local_files_only=False,
            allow_patterns=allow_patterns,
        )
        logger.debug(f"Downloaded from HF Hub: {downloaded}")
        return Path(downloaded)
    except Exception as e:
        logger.error(f"HuggingFace download failed for '{model_id}': {e}")
        return None


async def download_model(
    model_id: str,
    cache_dir: Path,
    tokenizer_only: bool = False,
) -> Path | None:
    hf_path = download_from_hf(model_id, cache_dir, tokenizer_only=tokenizer_only)
    if hf_path and validate_model_path(hf_path, tokenizer_only):
        logger.debug(f"Downloaded from HF Hub: {hf_path}")
        return hf_path

    return None
