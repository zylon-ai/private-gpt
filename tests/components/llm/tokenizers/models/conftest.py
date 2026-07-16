"""Shared fixtures for model cache/discovery/downloader tests."""

import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Cache Directory Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hf_cache_dir(tmp_path: Path) -> Path:
    """Create a temporary HuggingFace-style cache directory."""
    cache = tmp_path / "cache" / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


@pytest.fixture
def create_hf_repo(hf_cache_dir: Path):
    """Factory fixture to create HF-style repo directories."""

    def _create_repo(
        model_id: str,
        snapshots: list[dict[str, Any]] | None = None,
        root_files: list[str] | None = None,
    ) -> Path:
        """Create a HuggingFace repository structure.

        Args:
            model_id: Model ID like "mistralai/mistral-7b-v0.1"
            snapshots: List of snapshot configs like:
                [{"hash": "abc123", "mtime": 1000,
                  "files": ["tokenizer.json", ...]}, ...]
            root_files: List of files to create at repo root
        """
        repo_name = f"models--{model_id.replace('/', '--')}"
        repo_dir = hf_cache_dir / repo_name
        repo_dir.mkdir(parents=True, exist_ok=True)

        # Create root files
        if root_files:
            for filename in root_files:
                (repo_dir / filename).touch()

        # Create snapshots
        if snapshots:
            snapshot_dir = repo_dir / "snapshots"
            snapshot_dir.mkdir(exist_ok=True)

            for snap in snapshots:
                snap_hash = snap["hash"]
                snap_path = snapshot_dir / snap_hash
                snap_path.mkdir(exist_ok=True)

                # Create files in snapshot
                for filename in snap.get("files", []):
                    (snap_path / filename).touch()

                # Set mtime if specified
                if "mtime" in snap:
                    os.utime(snap_path, (snap["mtime"], snap["mtime"]))

        return repo_dir

    return _create_repo


@pytest.fixture
def tokenizer_files() -> list[str]:
    """Standard HuggingFace tokenizer files."""
    return ["tokenizer.json", "tokenizer_config.json"]


@pytest.fixture
def mistral_tokenizer_files() -> list[str]:
    """Mistral tekken tokenizer files."""
    return ["tekken.json"]


@pytest.fixture
def safetensors_files() -> list[str]:
    """Model weight files."""
    return ["model.safetensors", "model-00001-of-00002.safetensors"]


# ---------------------------------------------------------------------------
# Mock HuggingFace Hub
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_hf_hub_success(monkeypatch, hf_cache_dir: Path):
    """Mock huggingface_hub.snapshot_download that succeeds."""

    def mock_snapshot_download(
        repo_id: str,
        cache_dir: str,
        local_files_only: bool = False,
        allow_patterns: list[str] | None = None,
        **kwargs,
    ) -> str:
        # Create the model in cache
        repo_name = f"models--{repo_id.replace('/', '--')}"
        repo_dir = Path(cache_dir) / repo_name

        # Create snapshot
        snapshot_dir = repo_dir / "snapshots"
        snap_hash = "abc123"
        snap_path = snapshot_dir / snap_hash
        snap_path.mkdir(parents=True, exist_ok=True)

        # Create files based on allow_patterns
        if allow_patterns:
            # Tokenizer only
            (snap_path / "tokenizer.json").touch()
            (snap_path / "tokenizer_config.json").touch()
        else:
            # Full model
            (snap_path / "tokenizer.json").touch()
            (snap_path / "tokenizer_config.json").touch()
            (snap_path / "model.safetensors").touch()

        return str(snap_path)

    # Mock where it's imported and used (inside download_from_hf)
    import sys

    mock_hub = MagicMock()
    mock_hub.snapshot_download = mock_snapshot_download
    sys.modules["huggingface_hub"] = mock_hub

    return mock_snapshot_download


@pytest.fixture
def mock_hf_hub_failure(monkeypatch):
    """Mock huggingface_hub.snapshot_download that fails."""

    def mock_snapshot_download(*args, **kwargs) -> str:
        raise RuntimeError("Network error: cannot reach HuggingFace Hub")

    # Mock where it's imported
    import sys

    mock_hub = MagicMock()
    mock_hub.snapshot_download = mock_snapshot_download
    sys.modules["huggingface_hub"] = mock_hub

    return mock_snapshot_download


@pytest.fixture
def mock_hf_hub_not_available(monkeypatch):
    """Mock huggingface_hub as not installed."""
    import builtins
    import sys

    if "huggingface_hub" in sys.modules:
        monkeypatch.delitem(sys.modules, "huggingface_hub")

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError("huggingface_hub not available")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)


# ---------------------------------------------------------------------------
# Environment Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_env_offline_mode(monkeypatch):
    """Set environment for offline mode."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@pytest.fixture
def mock_env_model_repos(monkeypatch):
    """Factory to set MODEL_REPOS environment variable."""

    def _set_repos(repos: list[str]):
        monkeypatch.setenv("MODEL_REPOS", ",".join(repos))

    return _set_repos


# ---------------------------------------------------------------------------
# Broken State Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def broken_cache_incomplete_tokenizer(hf_cache_dir: Path):
    """Create cache with incomplete tokenizer (missing tokenizer_config.json)."""
    repo_name = "models--mistralai--mistral-7b-v0.1"
    repo_dir = hf_cache_dir / repo_name
    repo_dir.mkdir(parents=True, exist_ok=True)

    # Only tokenizer.json, missing tokenizer_config.json
    (repo_dir / "tokenizer.json").touch()

    return repo_dir


@pytest.fixture
def broken_cache_no_tokenizer(hf_cache_dir: Path):
    """Create cache with safetensors but no tokenizer."""
    repo_name = "models--mistralai--mistral-7b-v0.1"
    repo_dir = hf_cache_dir / repo_name
    repo_dir.mkdir(parents=True, exist_ok=True)

    # Only safetensors
    (repo_dir / "model.safetensors").touch()

    return repo_dir


@pytest.fixture
def broken_cache_empty_snapshot(hf_cache_dir: Path):
    """Create cache with empty snapshot directory."""
    repo_name = "models--mistralai--mistral-7b-v0.1"
    repo_dir = hf_cache_dir / repo_name
    snapshot_dir = repo_dir / "snapshots" / "empty_hash"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Root has valid files
    (repo_dir / "tokenizer.json").touch()
    (repo_dir / "tokenizer_config.json").touch()
    (repo_dir / "model.safetensors").touch()

    return repo_dir


@pytest.fixture
def broken_cache_corrupted_symlink(hf_cache_dir: Path):
    """Create cache with broken symlink."""
    repo_name = "models--mistralai--mistral-7b-v0.1"
    repo_dir = hf_cache_dir / repo_name
    repo_dir.mkdir(parents=True, exist_ok=True)

    # Create symlink to non-existent target
    symlink_path = repo_dir / "broken_link"
    symlink_path.symlink_to("/nonexistent/path")

    return repo_dir


# ---------------------------------------------------------------------------
# Timing Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def create_snapshots_with_timing(hf_cache_dir: Path):
    """Create snapshots with specific mtimes for ordering tests."""

    def _create(model_id: str, snapshots: list[tuple[str, float, list[str]]]) -> Path:
        repo_name = f"models--{model_id.replace('/', '--')}"
        repo_dir = hf_cache_dir / repo_name
        snapshot_dir = repo_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        now = time.time()
        for snap_hash, mtime_offset, files in snapshots:
            snap_path = snapshot_dir / snap_hash
            snap_path.mkdir(exist_ok=True)

            for filename in files:
                (snap_path / filename).touch()

            # Set mtime
            mtime = now + mtime_offset
            os.utime(snap_path, (mtime, mtime))

        return repo_dir

    return _create
