from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from io import TextIOWrapper

from private_gpt.components.llm.tokenizers.models.model_cache import (
    cleanup_stale_candidates,
    find_local_cache_model,
    find_local_model,
    set_model_permissions,
    validate_model_path,
)
from private_gpt.components.llm.tokenizers.models.model_downloader import download_model
from private_gpt.constants import PGPT_HOME

logger = logging.getLogger(__name__)

# CLI constants
HF_HOME = Path(os.environ.get("HF_HOME", str(PGPT_HOME / "models" / "cache")))
LOCK_FILE = HF_HOME / ".model-download.lock"
LOCK_TIMEOUT = int(os.environ.get("DOWNLOAD_LOCK_TIMEOUT", "3600"))
OFFLINE_MODE = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
JSON_OUTPUT = os.environ.get("MODEL_DOWNLOAD_JSON_OUTPUT", "0") == "1"


async def _discover_model(
    model_id: str,
    cache_dir: Path,
    force_download: bool,
    local_files_only: bool,
    tokenizer_only: bool,
) -> tuple[str, bool]:
    if not force_download:
        cache_path = find_local_cache_model(model_id, cache_dir, tokenizer_only)
        if cache_path:
            logger.info(f"Using HuggingFace cache: {cache_path}")
            return str(cache_path), True

    if local_files_only:
        logger.warning(
            f"local_files_only=True but model '{model_id}' not found locally"
        )
        return model_id, False

    logger.info(f"No cache found for '{model_id}' — attempting download")
    downloaded = await download_model(
        model_id=model_id,
        cache_dir=cache_dir,
        tokenizer_only=tokenizer_only,
    )
    if downloaded and validate_model_path(downloaded, tokenizer_only):
        logger.info(f"Downloaded model: {downloaded}")
        return str(downloaded), True

    logger.debug(f"Falling back to original identifier: {model_id}")
    return model_id, False


async def discover_model(
    model_id: str,
    cache_dir: Path,
    force_download: bool,
    local_files_only: bool,
    tokenizer_only: bool,
) -> tuple[str, bool]:
    """Async entry point for model resolution.

    Resolution order:
      1. HuggingFace hub cache
      2. Download from HuggingFace if not offline
      3. Falls back to original ``model_id``
    """
    try:
        resolved_id, is_local = await _discover_model(
            model_id=model_id,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            tokenizer_only=tokenizer_only,
        )
        if is_local:
            potential_path = Path(resolved_id)
            if potential_path.exists():
                cleanup_stale_candidates(
                    model_id=model_id,
                    keep_path=potential_path,
                    cache_dir=cache_dir,
                    tokenizer_only=tokenizer_only,
                )
                set_model_permissions(potential_path)

        return resolved_id, is_local
    except Exception as e:
        logger.exception(f"Error during model discovery for '{model_id}': {e}")

    return model_id, False


# ---------------------------------------------------------------------------
# File Lock (for CLI)
# ---------------------------------------------------------------------------


class FileLock:
    """File-based lock for coordinating downloads across multiple pods."""

    def __init__(self, lock_file: Path, timeout: int = LOCK_TIMEOUT) -> None:
        self.lock_file = lock_file
        self.timeout = timeout
        self._fd: TextIOWrapper | None = None

    def __enter__(self) -> FileLock:
        logger.info(f"Acquiring lock: {self.lock_file}")
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self._fd = open(self.lock_file, "w")
        start = time.time()
        while True:
            try:
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fd.write(f"{os.getpid()}\n")
                self._fd.flush()
                logger.info("Lock acquired")
                return self
            except OSError as e:
                elapsed = time.time() - start
                if elapsed >= self.timeout:
                    raise TimeoutError(
                        f"Failed to acquire lock within {self.timeout}s"
                    ) from e
                logger.info(f"Waiting for lock… ({elapsed:.0f}s / {self.timeout}s)")
                time.sleep(5)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if self._fd:
            try:
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
                self._fd.close()
                logger.info("Lock released")
            except Exception as e:
                logger.warning(f"Error releasing lock: {e}")


# ---------------------------------------------------------------------------
# CLI — Init Container Entry Point
# ---------------------------------------------------------------------------


async def is_model_cached(repo_id: str) -> tuple[bool, str | None]:
    """Check whether *repo_id* is available in any local cache."""
    try:
        path = await find_local_model(repo_id, str(HF_HOME))
        if path:
            return True, str(path)
        return False, None
    except Exception as e:
        logger.warning(f"Error checking cache for {repo_id}: {e}")
        return False, None


def get_model_repos() -> list[str]:
    """Parse MODEL_REPOS environment variable."""
    raw = os.environ.get("MODEL_REPOS", "")
    if not raw:
        return []
    repos = [r.strip() for r in raw.split(",") if r.strip()]
    return repos


async def _run(
    model_repos: list[str], json_output: bool = False
) -> dict[str, Any] | None:
    """Main download orchestration logic."""
    cached: dict[str, str] = {}
    to_download: list[str] = []
    missing_offline: list[str] = []

    if not json_output:
        print("Checking cache status...")
    for repo_id in model_repos:
        found, location = await is_model_cached(repo_id)
        if found and location is not None:
            cached[repo_id] = location
        elif OFFLINE_MODE:
            missing_offline.append(repo_id)
        else:
            to_download.append(repo_id)

    if not json_output:
        print("\n" + "=" * 80)
        print("Cache Status Summary")
        print("=" * 80)

    if not json_output:
        if cached:
            print(f"\n✓ Cached ({len(cached)}):")
            for repo_id, location in cached.items():
                print(f"  • {repo_id}  [{location}]")

        if to_download:
            print(f"\n⬇ To download ({len(to_download)}):")
            for repo_id in to_download:
                print(f"  • {repo_id}")

        if missing_offline:
            print(f"\n✗ Missing in offline mode ({len(missing_offline)}):")
            for repo_id in missing_offline:
                print(f"  • {repo_id}")

    if OFFLINE_MODE and missing_offline:
        if json_output:
            result = {
                "status": "error",
                "error": "Offline mode but models are missing",
                "cached": cached,
                "missing_offline": missing_offline,
            }
            return result
        else:
            print("\n" + "=" * 80)
            print("ERROR: Offline mode but models are missing")
            print("=" * 80)
            for repo_id in missing_offline:
                print(f"  ✗ {repo_id}")
            sys.exit(1)

    if not to_download:
        if json_output:
            return {
                "status": "success",
                "message": "All models cached",
                "cached": cached,
                "downloaded": [],
                "skipped": [],
                "failed": [],
            }
        else:
            print("\n✓ All models cached — no downloads needed\n")
            return None

    if not json_output:
        print("\n" + "=" * 80)
        print(f"Downloading {len(to_download)} model(s)")
        print("=" * 80)

    with FileLock(LOCK_FILE):
        downloaded: list[str] = []
        skipped: list[str] = []
        failed: list[str] = []

        for i, repo_id in enumerate(to_download, 1):
            if not json_output:
                print(f"\n[{i}/{len(to_download)}] {repo_id}")
                print("-" * 80)

            found, location = await is_model_cached(repo_id)
            if found:
                if not json_output:
                    print("✓ Already downloaded (by another pod)")
                    logger.debug(f"  Location: {location}")
                skipped.append(repo_id)
                if location:
                    cached[repo_id] = location
                continue

            local_path: Path | None = await download_model(repo_id, HF_HOME)
            if local_path:
                if not json_output:
                    print("✓ Downloaded successfully")
                    logger.debug(f"  Path: {local_path}")
                downloaded.append(repo_id)
                cached[repo_id] = str(local_path)

                cleanup_stale_candidates(
                    model_id=repo_id,
                    keep_path=local_path,
                    cache_dir=HF_HOME,
                    tokenizer_only=False,
                )
                set_model_permissions(local_path)

            else:
                if not json_output:
                    print("✗ Download failed")
                failed.append(repo_id)

        if json_output:
            result = {
                "status": "success" if not failed else "partial_failure",
                "cached": cached,
                "downloaded": downloaded,
                "skipped": skipped,
                "failed": failed,
            }
            return result
        else:
            print("\n" + "=" * 80)
            print("Summary")
            print("=" * 80)
            print(f"  Downloaded: {len(downloaded)}")
            print(f"  Skipped:    {len(skipped)}")
            print(f"  Failed:     {len(failed)}")

            if failed:
                print("\nFailed models:")
                for repo_id in failed:
                    print(f"  ✗ {repo_id}")
                sys.exit(1)

            print("\n✓ All models are now available\n")
            return None


def main() -> None:
    """CLI entry point for init container."""
    # Configure simple logging format for clean CLI output
    if JSON_OUTPUT:
        # Suppress all logging output when JSON mode is enabled
        logging.basicConfig(
            level=logging.ERROR,
            format="%(message)s",
            force=True,
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            force=True,
        )

    if not JSON_OUTPUT:
        print("\n" + "=" * 80)
        print("AI Model Downloader — Init Container")
        print("=" * 80)
        print(f"  HF_HOME:      {HF_HOME}")
        print(f"  Offline mode: {OFFLINE_MODE}")
        print(f"  Lock timeout: {LOCK_TIMEOUT}s")
        print("=" * 80 + "\n")

    HF_HOME.mkdir(parents=True, exist_ok=True)

    model_repos = get_model_repos()
    if not model_repos:
        if JSON_OUTPUT:
            print(
                json.dumps(
                    {"status": "success", "message": "No models to check", "cached": {}}
                )
            )
        else:
            print("MODEL_REPOS is empty — nothing to do")
        sys.exit(0)

    if not JSON_OUTPUT:
        print(f"Models to check ({len(model_repos)}):")
        for i, repo_id in enumerate(model_repos, 1):
            print(f"  {i}. {repo_id}")
        print()

    try:
        result = asyncio.run(_run(model_repos, json_output=JSON_OUTPUT))
        if JSON_OUTPUT and result:
            print(json.dumps(result, indent=2))
            # Exit with error code if there were failures
            if result.get("status") == "error" or result.get("failed"):
                sys.exit(1)
    except SystemExit:
        raise
    except TimeoutError as e:
        logger.error(f"Lock timeout: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
