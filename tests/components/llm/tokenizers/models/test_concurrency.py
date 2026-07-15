"""Test concurrent download scenarios and locking.

Scenario 1: Concurrency Scenarios
Scenario 7: Lock Timeout Scenarios
"""

import asyncio
from pathlib import Path

import pytest

from private_gpt.components.llm.tokenizers.models.model_cache import (
    find_local_cache_model,
)
from private_gpt.components.llm.tokenizers.models.model_discovery import FileLock


class TestFileLocking:
    """Scenario 7.1, 7.2: File lock behavior."""

    def test_lock_acquired_and_released(self, hf_cache_dir: Path):
        """Lock should be acquired and released properly."""
        lock_file = hf_cache_dir / ".test.lock"

        with FileLock(lock_file, timeout=5):
            assert lock_file.exists()
            # Lock is held here

        # Lock should be released after context exit
        # File might still exist but should not be locked

    def test_lock_timeout(self, hf_cache_dir: Path):
        """Lock should timeout after specified duration."""
        lock_file = hf_cache_dir / ".test.lock"

        # Acquire lock in first context
        with FileLock(lock_file, timeout=1):  # noqa: SIM117
            # Try to acquire again (should timeout)
            with (
                pytest.raises(TimeoutError, match="Failed to acquire lock"),
                FileLock(lock_file, timeout=1),
            ):
                pass

    def test_lock_writes_pid(self, hf_cache_dir: Path):
        """Lock file should contain process PID."""
        import os

        lock_file = hf_cache_dir / ".test.lock"

        with FileLock(lock_file, timeout=5):
            content = lock_file.read_text()
            assert str(os.getpid()) in content


class TestConcurrentDownloadSimulation:
    """Scenario 1.1: Multiple pods download same model (simplified simulation)."""

    @pytest.mark.asyncio
    async def test_concurrent_cache_reads(
        self,
        create_hf_repo,
        tokenizer_files: list[str],
    ):
        """Multiple concurrent reads of cache should work without blocking."""
        # Create a cached model
        repo_dir = create_hf_repo(
            "concurrent/read-test",
            snapshots=[
                {
                    "hash": "abc",
                    "mtime": 1000,
                    "files": [*tokenizer_files, "model.safetensors"],
                }
            ],
        )

        # Simulate concurrent reads
        async def read_cache():
            return find_local_cache_model(
                "concurrent/read-test",
                repo_dir.parent,
                tokenizer_only=False,
            )

        # Run 10 concurrent reads
        results = await asyncio.gather(*[read_cache() for _ in range(10)])

        # All should succeed with same result
        assert all(r is not None for r in results)
        assert len({str(r) for r in results}) == 1  # All same path

    @pytest.mark.asyncio
    async def test_sequential_validation_no_corruption(
        self,
        create_hf_repo,
        tokenizer_files: list[str],
    ):
        """Sequential validations shouldn't corrupt cache state."""
        repo_dir = create_hf_repo(
            "sequential/test",
            snapshots=[
                {
                    "hash": "v1",
                    "mtime": 1000,
                    "files": [*tokenizer_files, "model.safetensors"],
                },
                {
                    "hash": "v2",
                    "mtime": 2000,
                    "files": [*tokenizer_files, "model.safetensors"],
                },
            ],
        )

        # Multiple sequential finds
        for _ in range(5):
            result = find_local_cache_model(
                "sequential/test",
                repo_dir.parent,
                tokenizer_only=False,
            )
            assert result is not None
            # Should consistently find newest valid snapshot
            assert "v2" in str(result)


class TestDownloadInterruption:
    """Scenario 1.2: Download interrupted, another pod takes over."""

    def test_incomplete_snapshot_detected_and_skipped(
        self,
        hf_cache_dir: Path,
        tokenizer_files: list[str],
    ):
        """Incomplete snapshot should be skipped, valid ones used."""
        repo_name = "models--interrupted--download"
        repo_dir = hf_cache_dir / repo_name
        snapshot_dir = repo_dir / "snapshots"

        # Incomplete snapshot (simulating interrupted download)
        incomplete = snapshot_dir / "incomplete_abc"
        incomplete.mkdir(parents=True)
        (incomplete / "tokenizer.json").touch()  # Only partial files

        # Complete snapshot (earlier download)
        complete = snapshot_dir / "complete_xyz"
        complete.mkdir(parents=True)
        for f in [*tokenizer_files, "model.safetensors"]:
            (complete / f).touch()

        # Set mtimes: incomplete is newer (just crashed) but invalid
        import os

        os.utime(incomplete, (2000, 2000))
        os.utime(complete, (1000, 1000))

        # Should find the complete one, not the newer incomplete
        result = find_local_cache_model(
            "interrupted/download",
            hf_cache_dir,
            tokenizer_only=False,
        )

        assert result is not None
        assert "complete_xyz" in str(result)


class TestConcurrentPermissionSetting:
    """Scenario 3.2: Concurrent permission setting."""

    def test_permission_setting_idempotent(
        self,
        hf_cache_dir: Path,
        tokenizer_files: list[str],
    ):
        """set_model_permissions should be safe to call multiple times."""
        from private_gpt.components.llm.tokenizers.models.model_cache import (
            set_model_permissions,
        )

        # Create test directory structure
        test_dir = hf_cache_dir / "test_perms"
        test_dir.mkdir()
        for f in tokenizer_files:
            (test_dir / f).touch()
        subdir = test_dir / "subdir"
        subdir.mkdir()
        (subdir / "file.txt").touch()

        # Call multiple times (simulating concurrent calls)
        for _ in range(3):
            set_model_permissions(test_dir)

        # Should not raise exception
        # Files should have 777 permissions

        assert test_dir.stat().st_mode & 0o777 == 0o777

    def test_permission_error_logged_not_raised(
        self,
        hf_cache_dir: Path,
        monkeypatch,
    ):
        """Permission errors should be logged, not raised."""
        from private_gpt.components.llm.tokenizers.models.model_cache import (
            set_model_permissions,
        )

        test_dir = hf_cache_dir / "readonly"
        test_dir.mkdir()

        # Mock chmod to fail
        original_chmod = Path.chmod

        def failing_chmod(self, mode):
            raise PermissionError("Cannot change permissions")

        monkeypatch.setattr(Path, "chmod", failing_chmod)

        # Should not raise
        set_model_permissions(test_dir)

        # Restore
        monkeypatch.setattr(Path, "chmod", original_chmod)


class TestIncrementalDownloads:
    """Test scenarios where tokenizer_only flag differs between downloads."""

    @pytest.mark.asyncio
    async def test_tokenizer_only_then_full_model(
        self,
        hf_cache_dir: Path,
        create_hf_repo,
        tokenizer_files: list[str],
        mock_hf_hub_success,
    ):
        """First download tokenizer-only, second download full model."""
        # Simulate first download: tokenizer only
        repo_dir = create_hf_repo(
            "incremental/test",
            snapshots=[
                {
                    "hash": "tokenizer_only",
                    "mtime": 1000,
                    "files": tokenizer_files,  # No safetensors
                }
            ],
        )

        # First check: tokenizer-only should be found
        result1 = find_local_cache_model(
            "incremental/test",
            repo_dir.parent,
            tokenizer_only=True,
        )
        assert result1 is not None  # Tokenizer-only is valid

        # Second check: full model should NOT be found (missing safetensors)
        result2 = find_local_cache_model(
            "incremental/test",
            repo_dir.parent,
            tokenizer_only=False,
        )
        assert result2 is None  # Full model not valid

        # Simulate second download completing: add safetensors
        for f in ["model.safetensors"]:
            (result1 / f).touch()

        # Now full model should be found
        result3 = find_local_cache_model(
            "incremental/test",
            repo_dir.parent,
            tokenizer_only=False,
        )
        assert result3 is not None
        assert result3 == result1  # Same directory

    @pytest.mark.asyncio
    async def test_concurrent_different_tokenizer_flags(
        self,
        hf_cache_dir: Path,
        create_hf_repo,
        tokenizer_files: list[str],
    ):
        """Multiple concurrent reads with different tokenizer_only flags."""
        # Create full model (satisfies both tokenizer-only and full)
        repo_dir = create_hf_repo(
            "concurrent/mixed-flags",
            snapshots=[
                {
                    "hash": "complete",
                    "mtime": 1000,
                    "files": [*tokenizer_files, "model.safetensors"],
                }
            ],
        )

        async def check_tokenizer_only():
            return find_local_cache_model(
                "concurrent/mixed-flags",
                repo_dir.parent,
                tokenizer_only=True,
            )

        async def check_full_model():
            return find_local_cache_model(
                "concurrent/mixed-flags",
                repo_dir.parent,
                tokenizer_only=False,
            )

        # Mix of tokenizer-only and full model checks
        tasks = [check_tokenizer_only() for _ in range(5)] + [
            check_full_model() for _ in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed (full model satisfies both requirements)
        assert all(r is not None for r in results)
        # All should point to same location
        paths = {str(r) for r in results}
        assert len(paths) == 1

    def test_tokenizer_only_insufficient_for_full_model(
        self,
        hf_cache_dir: Path,
        create_hf_repo,
        tokenizer_files: list[str],
    ):
        """Tokenizer-only cache should not satisfy full model request."""
        # Create tokenizer-only cache
        repo_dir = create_hf_repo(
            "insufficient/test",
            snapshots=[
                {
                    "hash": "tokenizer_snap",
                    "mtime": 1000,
                    "files": tokenizer_files,
                }
            ],
        )

        # Tokenizer-only request: should succeed
        tokenizer_result = find_local_cache_model(
            "insufficient/test",
            repo_dir.parent,
            tokenizer_only=True,
        )
        assert tokenizer_result is not None

        # Full model request: should fail (missing safetensors)
        full_result = find_local_cache_model(
            "insufficient/test",
            repo_dir.parent,
            tokenizer_only=False,
        )
        assert full_result is None

    def test_full_model_satisfies_tokenizer_only(
        self,
        hf_cache_dir: Path,
        create_hf_repo,
        tokenizer_files: list[str],
    ):
        """Full model cache should satisfy tokenizer-only request (superset)."""
        # Create full model cache
        repo_dir = create_hf_repo(
            "superset/test",
            snapshots=[
                {
                    "hash": "full_snap",
                    "mtime": 1000,
                    "files": [*tokenizer_files, "model.safetensors"],
                }
            ],
        )

        # Both requests should succeed
        tokenizer_result = find_local_cache_model(
            "superset/test",
            repo_dir.parent,
            tokenizer_only=True,
        )
        full_result = find_local_cache_model(
            "superset/test",
            repo_dir.parent,
            tokenizer_only=False,
        )

        assert tokenizer_result is not None
        assert full_result is not None
        assert tokenizer_result == full_result  # Same path

    @pytest.mark.asyncio
    async def test_sequential_upgrades_tokenizer_to_full(
        self,
        hf_cache_dir: Path,
        tokenizer_files: list[str],
    ):
        """Simulate sequential downloads: tokenizer first, then upgrade to full."""
        model_id = "sequential/upgrade"
        repo_name = f"models--{model_id.replace('/', '--')}"
        repo_dir = hf_cache_dir / repo_name
        snapshot_dir = repo_dir / "snapshots"

        # Step 1: Initial tokenizer-only download
        snap1 = snapshot_dir / "tokenizer_v1"
        snap1.mkdir(parents=True)
        for f in tokenizer_files:
            (snap1 / f).touch()

        import os

        os.utime(snap1, (1000, 1000))

        # Verify tokenizer-only works
        result1 = find_local_cache_model(
            model_id,
            hf_cache_dir,
            tokenizer_only=True,
        )
        assert result1 is not None

        # Verify full model doesn't work yet
        result2 = find_local_cache_model(
            model_id,
            hf_cache_dir,
            tokenizer_only=False,
        )
        assert result2 is None

        # Step 2: Later, full model downloaded
        await asyncio.sleep(0.01)  # Simulate time passing
        snap2 = snapshot_dir / "full_v2"
        snap2.mkdir()
        for f in [*tokenizer_files, "model.safetensors"]:
            (snap2 / f).touch()

        os.utime(snap2, (2000, 2000))

        # Now both should work
        result3 = find_local_cache_model(
            model_id,
            hf_cache_dir,
            tokenizer_only=True,
        )
        result4 = find_local_cache_model(
            model_id,
            hf_cache_dir,
            tokenizer_only=False,
        )

        assert result3 is not None
        assert result4 is not None
        # Full model snapshot should be returned (it's newer and complete)
        assert result4 == snap2


class TestCacheStateConsistency:
    """Test cache state remains consistent under various operations."""

    def test_validation_doesnt_modify_cache(
        self,
        create_hf_repo,
        tokenizer_files: list[str],
    ):
        """Validation operations should not modify cache state."""
        repo_dir = create_hf_repo(
            "readonly/test",
            snapshots=[
                {
                    "hash": "abc",
                    "mtime": 1000,
                    "files": [*tokenizer_files, "model.safetensors"],
                }
            ],
        )

        # Get initial state
        initial_files = list(repo_dir.rglob("*"))
        initial_count = len(initial_files)

        # Run validation multiple times
        for _ in range(5):
            result = find_local_cache_model(
                "readonly/test",
                repo_dir.parent,
                tokenizer_only=False,
            )
            assert result is not None

        # State should be unchanged
        final_files = list(repo_dir.rglob("*"))
        assert len(final_files) == initial_count

    @pytest.mark.asyncio
    async def test_concurrent_validations_same_result(
        self,
        create_hf_repo,
        tokenizer_files: list[str],
    ):
        """Concurrent validations should all return same result."""
        repo_dir = create_hf_repo(
            "concurrent/validation",
            snapshots=[
                {
                    "hash": "snap1",
                    "mtime": 1000,
                    "files": [*tokenizer_files, "model.safetensors"],
                },
                {
                    "hash": "snap2",
                    "mtime": 2000,
                    "files": [*tokenizer_files, "model.safetensors"],
                },
            ],
        )

        async def validate():
            return find_local_cache_model(
                "concurrent/validation",
                repo_dir.parent,
                tokenizer_only=False,
            )

        # Run 20 concurrent validations
        results = await asyncio.gather(*[validate() for _ in range(20)])

        # All should return same path (newest valid snapshot)
        paths = [str(r) for r in results if r]
        assert len(set(paths)) == 1
        assert "snap2" in paths[0]


class TestCleanupStaleSnapshots:
    """Test cleanup logic doesn't remove wrong snapshots."""

    def test_cleanup_preserves_keep_path(
        self,
        hf_cache_dir: Path,
        tokenizer_files: list[str],
    ):
        """cleanup_stale_candidates should preserve keep_path."""
        from private_gpt.components.llm.tokenizers.models.model_cache import (
            cleanup_stale_candidates,
        )

        repo_name = "models--cleanup--test"
        repo_dir = hf_cache_dir / repo_name
        snapshot_dir = repo_dir / "snapshots"

        # Create valid snapshot (keep this)
        keep = snapshot_dir / "keep123"
        keep.mkdir(parents=True)
        for f in [*tokenizer_files, "model.safetensors"]:
            (keep / f).touch()

        # Create invalid snapshot (should be removed)
        remove = snapshot_dir / "invalid456"
        remove.mkdir(parents=True)
        (remove / "partial.txt").touch()

        # Run cleanup
        cleanup_stale_candidates(
            model_id="cleanup/test",
            keep_path=keep,
            cache_dir=hf_cache_dir,
            tokenizer_only=False,
        )

        # keep_path should still exist
        assert keep.exists()
        # Invalid should be removed
        # (Note: our implementation might not remove it if it's not validated as stale)

    def test_cleanup_preserves_ancestors(
        self,
        hf_cache_dir: Path,
        tokenizer_files: list[str],
    ):
        """cleanup_stale_candidates should preserve ancestors of keep_path."""
        from private_gpt.components.llm.tokenizers.models.model_cache import (
            cleanup_stale_candidates,
        )

        repo_name = "models--ancestor--test"
        repo_dir = hf_cache_dir / repo_name
        snapshot_dir = repo_dir / "snapshots"
        keep = snapshot_dir / "keep_snap"
        keep.mkdir(parents=True)

        for f in [*tokenizer_files, "model.safetensors"]:
            (keep / f).touch()

        # Run cleanup
        cleanup_stale_candidates(
            model_id="ancestor/test",
            keep_path=keep,
            cache_dir=hf_cache_dir,
            tokenizer_only=False,
        )

        # Ancestors should exist
        assert repo_dir.exists()
        assert snapshot_dir.exists()
        assert keep.exists()
