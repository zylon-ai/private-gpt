from __future__ import annotations

from typing import TYPE_CHECKING

from private_gpt.components.code_execution.bash_executor import LocalBashExecutor
from private_gpt.components.sandbox.local import LocalSandboxProvider
from private_gpt.components.web.scraper.pool import (
    PooledWebScraperProvider,
    ScrapeSessionPool,
)

if TYPE_CHECKING:
    from private_gpt.settings.settings import Settings

# Chromium needs far more headroom than the `settings.bash` defaults
# (512MB RLIMIT_AS / 50 RLIMIT_NPROC would kill it on Linux). RLIMIT_NPROC
# counts ALL processes of the user, not just this subtree, so it must clear
# the baseline process count of a busy host (e.g. a developer machine).
_SCRAPE_CPU_LIMIT_SECONDS = 120
_SCRAPE_MEMORY_LIMIT_MB = 4096
_SCRAPE_FSIZE_LIMIT_MB = 256
_SCRAPE_NPROC_LIMIT = 4096


class LocalWebScraperProvider(PooledWebScraperProvider):
    """Runs the scrape script on the host through the local sandbox executor.

    Uses private-gpt's own interpreter, so the `tool-web-scraping` extra
    (playwright) and installed chromium browsers serve this provider.
    """

    _BASE_DIR = "/home/agent/"

    def __init__(self, settings: Settings) -> None:
        executor = LocalBashExecutor(
            cpu_limit_seconds=_SCRAPE_CPU_LIMIT_SECONDS,
            memory_limit_mb=_SCRAPE_MEMORY_LIMIT_MB,
            fsize_limit_mb=_SCRAPE_FSIZE_LIMIT_MB,
            nproc_limit=_SCRAPE_NPROC_LIMIT,
            output_cap_bytes=settings.bash.output_cap_bytes,
        )
        pool = ScrapeSessionPool(
            LocalSandboxProvider(settings, executor=executor),
            pool_size=settings.web_fetch.pool_size,
            max_requests_per_session=settings.web_fetch.max_requests_per_session,
            idle_timeout_seconds=settings.web_fetch.pool_idle_timeout_seconds,
            user_id="web-scraper",
        )
        super().__init__(settings, pool, self._BASE_DIR)
