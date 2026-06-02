"""Model Cache Utilities.

All utilities for finding, checking, and managing cached models.
No downloading logic - pure cache inspection.
"""

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def default_cache_dir(cache_dir: str | None = None) -> Path:
    """Get the default cache directory path."""
    if cache_dir:
        return Path(cache_dir)
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    return Path(hf_home) / "hub"


# ---------------------------------------------------------------------------
# Cache Inspection Utilities
# ---------------------------------------------------------------------------


def _has_safetensors(path: Path) -> bool:
    """Check if a directory contains .safetensors files."""
    return any(path.glob("*.safetensors"))


def has_all_safetensors(path: Path) -> bool:
    """Check if a directory contains only .safetensors files (no .bin)."""
    if not _has_safetensors(path):
        return False

    safetensors_index_path = path / "model.safetensors.index.json"
    if safetensors_index_path.exists():
        try:
            import json

            with safetensors_index_path.open() as f:
                index_data = json.load(f)

            required_weight_files = set(index_data.get("weight_map", {}).values())
            downloaded_weight_files = {f.name for f in path.glob("*") if f.is_file()}

            if not required_weight_files.issubset(downloaded_weight_files):
                logger.warning(
                    f"Missing weight files for {path}: "
                    f"required {required_weight_files}, found {downloaded_weight_files}"
                )
                return False
            return True
        except Exception as e:
            logger.warning(f"Failed to parse {safetensors_index_path}: {e}")

    return True


def has_tokenizer_files(path: Path) -> bool:
    """Check if a directory contains tokenizer files."""
    tokenizer_files = [
        # Standard HuggingFace tokenizer files
        [
            "tokenizer.json",
            "tokenizer_config.json",
        ],
        # Mistral's tekken tokenizer
        [
            "tekken.json",
        ],
    ]

    for group in tokenizer_files:
        if all((path / filename).exists() for filename in group):
            return True
    return False


def validate_model_path(path: Path, tokenizer_only: bool) -> bool:
    """Validate if a path contains required files based on tokenizer_only flag.

    Args:
        path: Directory path to validate
        tokenizer_only: If True, only check for tokenizer files.
                       If False, check for both tokenizer files and safetensors.

    Returns:
        True if path contains required files, False otherwise.
    """
    if not path.exists() or not path.is_dir():
        return False

    has_tokenizer = has_tokenizer_files(path)
    has_weights = has_all_safetensors(path)

    if tokenizer_only:
        return has_tokenizer
    else:
        # For full model: need tokenizer AND weights
        return has_tokenizer and has_weights


def find_base_candidates(base_path: Path, hf_model_id: str) -> list[Path]:
    """Find all candidate directories for a HuggingFace model ID."""
    candidates: list[Path] = []
    seen: set[Path] = set()
    hf_model_id_lower = hf_model_id.lower()

    for name in (hf_model_id, hf_model_id_lower):
        candidate = base_path / name
        if candidate.exists() and candidate.is_dir():
            resolved = candidate.resolve()
            if resolved not in seen:
                candidates.append(candidate)
                seen.add(resolved)

    if base_path.exists():
        for item in base_path.iterdir():
            if (
                item.is_dir()
                and item.name.lower() == hf_model_id_lower
                and item.resolve() not in seen
            ):
                candidates.append(item)
                seen.add(item.resolve())

    return candidates


def find_repo_candidates(base_path: Path, hf_model_id: str) -> list[Path]:
    """Find all repository candidate directories for a HuggingFace model ID."""
    dir_candidates = find_base_candidates(base_path, hf_model_id)

    seen: set[tuple[int, int]] = set()
    all_candidates: list[Path] = []

    def _add(path: Path) -> bool:
        """Return True and register path if it hasn't been seen yet."""
        try:
            st = path.stat()
            key = (st.st_dev, st.st_ino)
        except OSError:
            return False
        if key in seen:
            return False
        seen.add(key)
        return True

    for candidate in dir_candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        if candidate.is_symlink():
            logger.debug(f"Skipping symlink candidate: {candidate}")
            continue
        if not _add(candidate):
            continue

        snapshot_dir = candidate / "snapshots"
        if snapshot_dir.exists() and snapshot_dir.is_dir():
            snapshots = sorted(
                (d for d in snapshot_dir.iterdir() if d.is_dir()),
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )
            for snap in snapshots:
                if _add(snap):
                    all_candidates.append(snap)

        all_candidates.append(candidate)

    return all_candidates


def find_local_cache_model(
    repo_id: str,
    base_path: Path,
    tokenizer_only: bool = False,
) -> Path | None:
    """Find a model in the HuggingFace cache directory."""
    hf_model_id = (
        repo_id if "models--" in repo_id else "models--" + repo_id.replace("/", "--")
    )

    all_candidates = find_repo_candidates(base_path, hf_model_id)
    for candidate in all_candidates:
        if validate_model_path(candidate, tokenizer_only):
            logger.debug(f"Found valid model at: {candidate}")
            return candidate

    return None


async def find_local_model(
    model_id: str,
    cache_dir: str | None = None,
    tokenizer_only: bool = False,
) -> Path | None:
    """Async local-only cache lookup.

    This is a pure cache inspection function - no downloads, no network calls.
    Used by the downloader CLI and service backends.
    """
    try:
        base_path = default_cache_dir(cache_dir)

        cache_path = find_local_cache_model(model_id, base_path, tokenizer_only)
        if cache_path:
            logger.debug(f"Model found in HF cache: {cache_path}")
            return cache_path

        return None

    except Exception as e:
        logger.warning(f"Error finding local model: {e}")
        return None


async def configure_model_path(
    model_id: str,
    cache_dir: str | None = None,
    tokenizer_only: bool = False,
) -> tuple[str, bool]:
    """Async path resolver for offline-only service backends (Triton, ZGPT).

    Returns:
        ``(resolved_path, is_local)``
    """
    base_path = default_cache_dir(cache_dir)
    try:
        cache_path = find_local_cache_model(model_id, base_path, tokenizer_only)
        if cache_path:
            logger.debug(f"Model found in HF cache: {cache_path}")
            return str(cache_path), True

        logger.warning(
            f"Model '{model_id}' not found in local cache. "
            f"Ensure the init container has completed successfully."
        )
        return model_id, False

    except Exception as e:
        logger.warning(f"Error resolving model path: {e}")
        return model_id, False
    finally:
        set_model_permissions(base_path)


# ---------------------------------------------------------------------------
# Download Helpers
# ---------------------------------------------------------------------------


def set_model_permissions(path: Path) -> None:
    """Recursively apply 777 so Triton and ZGPT (different Linux users) can read."""
    try:
        for p in [path, *path.rglob("*")]:
            p.chmod(0o777)
    except Exception as e:
        logger.warning(f"Failed to set permissions on {path}: {e}")


def cleanup_stale_candidates(
    model_id: str,
    keep_path: Path,
    cache_dir: Path,
    tokenizer_only: bool,
) -> None:
    """Remove stale/invalid candidate directories for *model_id*, keeping *keep_path*.

    Any directory that is keep_path itself, or an ancestor of it, is preserved.
    All other candidates that fail validation are removed.
    """
    hf_model_id = "models--" + model_id.replace("/", "--")
    keep_resolved = keep_path.resolve()

    for repo_dir in find_repo_candidates(cache_dir, hf_model_id):
        if not repo_dir.exists():
            continue
        if not repo_dir.is_dir():
            continue

        repo_resolved = repo_dir.resolve()

        if keep_resolved.is_relative_to(repo_resolved):
            logger.debug(f"Keeping ancestor of valid path: {repo_dir}")
            continue

        if not validate_model_path(repo_dir, tokenizer_only=tokenizer_only):
            logger.debug(f"Removing stale directory: {repo_dir}")
            shutil.rmtree(repo_dir, ignore_errors=True)
