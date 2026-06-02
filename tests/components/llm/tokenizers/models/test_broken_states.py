"""Test broken and corrupted cache states.

Scenario 2: Broken/Corrupted State Scenarios
"""
from pathlib import Path

from private_gpt.components.llm.tokenizers.models.model_cache import (
    find_local_cache_model,
    find_repo_candidates,
    validate_model_path,
)


class TestIncompleteCacheStates:
    """Scenario 2.1, 2.2: Test incomplete tokenizer and model states."""

    def test_incomplete_tokenizer_rejected(
        self, broken_cache_incomplete_tokenizer: Path
    ):
        """Incomplete tokenizer (missing tokenizer_config.json) → validation fails."""
        assert (
            validate_model_path(broken_cache_incomplete_tokenizer, tokenizer_only=True)
            is False
        )

    def test_safetensors_without_tokenizer_rejected(
        self, broken_cache_no_tokenizer: Path
    ):
        """Safetensors present, no tokenizer → validation fails."""
        assert (
            validate_model_path(broken_cache_no_tokenizer, tokenizer_only=False)
            is False
        )

    def test_find_local_cache_rejects_incomplete(
        self, hf_cache_dir: Path, broken_cache_incomplete_tokenizer: Path
    ):
        """Scenario 2.1: find_local_cache_model rejects incomplete tokenizer."""
        result = find_local_cache_model(
            "mistralai/mistral-7b-v0.1",
            hf_cache_dir,
            tokenizer_only=True,
        )
        assert result is None


class TestEmptySnapshots:
    """Scenario 2.3: Empty snapshot directory handling."""

    def test_empty_snapshot_skipped(self, broken_cache_empty_snapshot: Path):
        """Empty snapshot/ directory should be skipped, root with valid files used."""
        # Find candidates should return both snapshot and root
        candidates = find_repo_candidates(
            broken_cache_empty_snapshot.parent,
            broken_cache_empty_snapshot.name,
        )

        # Should find at least the root
        assert len(candidates) > 0

        # Validate the cache - should use root, not empty snapshot
        result = find_local_cache_model(
            "mistralai/mistral-7b-v0.1",
            broken_cache_empty_snapshot.parent,
            tokenizer_only=False,
        )

        # Should find the root directory with valid files
        assert result is not None
        assert result == broken_cache_empty_snapshot


class TestCorruptedSymlinks:
    """Scenario 2.4: Corrupted symlink handling."""

    def test_broken_symlink_skipped(self, broken_cache_corrupted_symlink: Path):
        """Broken symlinks should not cause exceptions."""
        # This should not raise, even with broken symlink
        candidates = find_repo_candidates(
            broken_cache_corrupted_symlink.parent,
            broken_cache_corrupted_symlink.name,
        )

        # Should still find the root directory
        assert broken_cache_corrupted_symlink in candidates


class TestCaseInsensitiveMatching:
    """Scenario 2.5: Case-insensitive model ID matching."""

    def test_lowercase_model_id_found(self, create_hf_repo, tokenizer_files: list[str]):
        """Model ID case mismatch should still find model."""
        # Create repo with mixed case
        repo_dir = create_hf_repo(
            "MistralAI/Mistral-7B-v0.1",
            snapshots=[
                {
                    "hash": "abc123",
                    "mtime": 1000,
                    "files": [*tokenizer_files, "model.safetensors"],
                }
            ],
        )

        # Request with different case
        result = find_local_cache_model(
            "mistralai/mistral-7b-v0.1",  # lowercase
            repo_dir.parent,
            tokenizer_only=False,
        )

        # Should find despite case mismatch
        assert result is not None

    def test_uppercase_model_id_found(self, create_hf_repo, tokenizer_files: list[str]):
        """Uppercase request should find lowercase cache."""
        repo_dir = create_hf_repo(
            "mistralai/mistral-7b-v0.1",  # lowercase in cache
            snapshots=[
                {
                    "hash": "def456",
                    "mtime": 2000,
                    "files": [*tokenizer_files, "model.safetensors"],
                }
            ],
        )

        # Request with uppercase (will be converted internally)
        result = find_local_cache_model(
            "mistralai/mistral-7b-v0.1",
            repo_dir.parent,
            tokenizer_only=False,
        )

        assert result is not None


class TestMixedValidityModels:
    """Scenario 5.2: Multiple models with mixed validity."""

    def test_finds_valid_among_invalid(
        self,
        hf_cache_dir: Path,
        tokenizer_files: list[str],
        safetensors_files: list[str],
    ):
        """Should find valid model and reject invalid ones."""
        # Create 3 different states
        model_id = "test/model"
        repo_name = f"models--{model_id.replace('/', '--')}"

        # Invalid: incomplete tokenizer
        invalid1 = hf_cache_dir / f"{repo_name}-invalid1"
        invalid1.mkdir()
        (invalid1 / "tokenizer.json").touch()  # missing tokenizer_config.json

        # Invalid: no tokenizer
        invalid2 = hf_cache_dir / f"{repo_name}-invalid2"
        invalid2.mkdir()
        (invalid2 / "model.safetensors").touch()  # no tokenizer

        # Valid: complete model
        valid = hf_cache_dir / repo_name
        valid.mkdir()
        for f in tokenizer_files + safetensors_files:
            (valid / f).touch()

        # Should find the valid one
        result = find_local_cache_model(model_id, hf_cache_dir, tokenizer_only=False)
        assert result is not None
        assert result == valid


class TestPartialDownloads:
    """Test handling of interrupted/partial downloads."""

    def test_partial_download_rejected(self, hf_cache_dir: Path):
        """Partial download (some files but not complete) should be rejected."""
        # Simulate partial download
        repo_name = "models--test--partial"
        repo_dir = hf_cache_dir / repo_name
        snap_dir = repo_dir / "snapshots" / "partial123"
        snap_dir.mkdir(parents=True)

        # Only partial files
        (snap_dir / "config.json").touch()
        # Missing tokenizer and safetensors

        result = find_local_cache_model(
            "test/partial", hf_cache_dir, tokenizer_only=False
        )
        assert result is None

    def test_partial_tokenizer_accepted_with_flag(
        self, hf_cache_dir: Path, tokenizer_files: list[str]
    ):
        """Partial download with complete tokenizer should work if tokenizer_only."""
        repo_name = "models--test--tokenizer-partial"
        repo_dir = hf_cache_dir / repo_name
        snap_dir = repo_dir / "snapshots" / "tok123"
        snap_dir.mkdir(parents=True)

        # Complete tokenizer files
        for f in tokenizer_files:
            (snap_dir / f).touch()

        # Missing safetensors (partial download)
        # But with tokenizer_only=True, should be valid
        result = find_local_cache_model(
            "test/tokenizer-partial", hf_cache_dir, tokenizer_only=True
        )
        assert result is not None
        assert result == snap_dir
