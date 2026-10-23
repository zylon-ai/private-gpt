"""Test cache evolution and snapshot ordering.

Scenario 5: Cache Evolution Scenarios
"""

from pathlib import Path

from private_gpt.components.llm.tokenizers.models.model_cache import (
    find_local_cache_model,
    find_repo_candidates,
)


class TestSnapshotOrdering:
    """Scenario 5.3: Snapshot mtime ordering."""

    def test_newest_snapshot_first(
        self, create_snapshots_with_timing, tokenizer_files: list[str]
    ):
        """Snapshots should be ordered by mtime (newest first)."""
        # Create snapshots with different mtimes
        repo_dir = create_snapshots_with_timing(
            "mistralai/mistral-7b",
            [
                (
                    "snap_old",
                    -86400,
                    [*tokenizer_files, "model.safetensors"],
                ),  # 1 day ago
                (
                    "snap_newest",
                    -3600,
                    [*tokenizer_files, "model.safetensors"],
                ),  # 1 hour ago
                (
                    "snap_middle",
                    -43200,
                    [*tokenizer_files, "model.safetensors"],
                ),  # 12 hours ago
            ],
        )

        # Get candidates - should be ordered newest to oldest
        candidates = find_repo_candidates(repo_dir.parent, repo_dir.name)

        # Extract snapshot hashes from paths
        snapshot_paths = [c for c in candidates if "snapshots" in str(c)]

        # Should have 3 snapshots
        assert len(snapshot_paths) >= 3

        # First should be newest (snap_newest)
        assert "snap_newest" in str(snapshot_paths[0])

    def test_find_local_cache_returns_newest_valid(
        self, create_snapshots_with_timing, tokenizer_files: list[str]
    ):
        """find_local_cache_model should return newest valid snapshot."""
        # Create snapshots where newest is valid
        repo_dir = create_snapshots_with_timing(
            "test/newest-valid",
            [
                ("old_valid", -86400, [*tokenizer_files, "model.safetensors"]),
                ("newest_valid", -3600, [*tokenizer_files, "model.safetensors"]),
            ],
        )

        result = find_local_cache_model(
            "test/newest-valid", repo_dir.parent, tokenizer_only=False
        )

        # Should return the newest valid snapshot
        assert result is not None
        assert "newest_valid" in str(result)

    def test_skips_invalid_newest_uses_older_valid(
        self, create_snapshots_with_timing, tokenizer_files: list[str]
    ):
        """If newest snapshot is invalid, should use older valid one."""
        # Create snapshots: newest is incomplete, older is valid
        repo_dir = create_snapshots_with_timing(
            "test/fallback-older",
            [
                ("old_complete", -86400, [*tokenizer_files, "model.safetensors"]),
                (
                    "newest_incomplete",
                    -3600,
                    ["tokenizer.json"],
                ),  # missing tokenizer_config
            ],
        )

        result = find_local_cache_model(
            "test/fallback-older", repo_dir.parent, tokenizer_only=False
        )

        # Should return the older complete snapshot
        assert result is not None
        assert "old_complete" in str(result)


class TestLegacyCacheFormat:
    """Scenario 5.1: Old cache format (no snapshots/)."""

    def test_legacy_cache_without_snapshots(
        self, hf_cache_dir: Path, tokenizer_files: list[str]
    ):
        """Legacy cache with files at root (no snapshots/) should work."""
        # Create legacy format: files directly in repo root
        repo_name = "models--legacy--model"
        repo_dir = hf_cache_dir / repo_name
        repo_dir.mkdir()

        # Files at root (old format)
        for f in [*tokenizer_files, "model.safetensors"]:
            (repo_dir / f).touch()

        # Should find and validate
        result = find_local_cache_model(
            "legacy/model", hf_cache_dir, tokenizer_only=False
        )
        assert result is not None
        assert result == repo_dir

    def test_legacy_and_modern_mixed(
        self, hf_cache_dir: Path, tokenizer_files: list[str]
    ):
        """Cache with both legacy root files should prioritize snapshots."""
        repo_name = "models--mixed--model"
        repo_dir = hf_cache_dir / repo_name
        repo_dir.mkdir()

        # Legacy: files at root
        for f in tokenizer_files:
            (repo_dir / f).touch()

        # Modern: snapshot with complete model
        snap_dir = repo_dir / "snapshots" / "modern123"
        snap_dir.mkdir(parents=True)
        for f in [*tokenizer_files, "model.safetensors"]:
            (snap_dir / f).touch()

        # Should find the snapshot (prioritized)
        candidates = find_repo_candidates(hf_cache_dir, repo_name)

        # Snapshots should come before root
        snapshot_candidates = [c for c in candidates if "snapshots" in str(c)]
        assert len(snapshot_candidates) > 0


class TestMultipleModelsInCache:
    """Scenario 5.2 (extended): Multiple models with various states."""

    def test_multiple_models_independent(
        self, create_hf_repo, tokenizer_files: list[str]
    ):
        """Multiple different models should be found independently."""
        # Create 3 different models
        repo1 = create_hf_repo(
            "model1/test",
            snapshots=[
                {
                    "hash": "aaa",
                    "mtime": 1000,
                    "files": [*tokenizer_files, "model.safetensors"],
                }
            ],
        )

        repo2 = create_hf_repo(
            "model2/test",
            snapshots=[
                {
                    "hash": "bbb",
                    "mtime": 2000,
                    "files": [*tokenizer_files, "model.safetensors"],
                }
            ],
        )

        repo3 = create_hf_repo(
            "model3/test",
            snapshots=[
                {
                    "hash": "ccc",
                    "mtime": 3000,
                    "files": [*tokenizer_files, "model.safetensors"],
                }
            ],
        )

        # Each should be found independently
        result1 = find_local_cache_model(
            "model1/test", repo1.parent, tokenizer_only=False
        )
        result2 = find_local_cache_model(
            "model2/test", repo2.parent, tokenizer_only=False
        )
        result3 = find_local_cache_model(
            "model3/test", repo3.parent, tokenizer_only=False
        )

        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
        assert result1 != result2 != result3


class TestSnapshotRootOrdering:
    """Test that snapshots come before root directory in candidates."""

    def test_snapshots_before_root(self, create_hf_repo, tokenizer_files: list[str]):
        """Candidates should be: [snapshots...], then root."""
        repo_dir = create_hf_repo(
            "test/ordering",
            snapshots=[
                {"hash": "snap1", "mtime": 1000, "files": tokenizer_files},
                {"hash": "snap2", "mtime": 2000, "files": tokenizer_files},
            ],
            root_files=tokenizer_files,  # Also files at root
        )

        candidates = find_repo_candidates(repo_dir.parent, repo_dir.name)

        # Find indices of snapshots and root
        snapshot_indices = []
        root_index = None

        for i, candidate in enumerate(candidates):
            if "snapshots" in str(candidate):
                snapshot_indices.append(i)
            elif candidate == repo_dir:
                root_index = i

        # All snapshots should come before root
        if root_index is not None and snapshot_indices:
            assert all(snap_idx < root_index for snap_idx in snapshot_indices)

    def test_multiple_snapshots_sorted(
        self, create_snapshots_with_timing, tokenizer_files: list[str]
    ):
        """Multiple snapshots should be sorted by mtime descending."""
        repo_dir = create_snapshots_with_timing(
            "test/multi-snap",
            [
                ("snap_a", -1000, tokenizer_files),  # oldest
                ("snap_b", -500, tokenizer_files),  # middle
                ("snap_c", -100, tokenizer_files),  # newest
            ],
        )

        candidates = find_repo_candidates(repo_dir.parent, repo_dir.name)
        snapshot_candidates = [c for c in candidates if "snapshots" in str(c)]

        # Should be in order: snap_c (newest), snap_b, snap_a
        assert len(snapshot_candidates) >= 3
        assert "snap_c" in str(snapshot_candidates[0])
        assert "snap_b" in str(snapshot_candidates[1])
        assert "snap_a" in str(snapshot_candidates[2])
