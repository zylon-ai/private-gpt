import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from private_gpt.components.web.web_search.models import WebSearchResult
from private_gpt.components.web.web_search.providers.base import BaseWebSearchProvider

logger = logging.getLogger(__name__)


class CachedProvider(BaseWebSearchProvider):
    """Wrapper que añade caché basada en ficheros con TTL a cualquier provider."""

    def __init__(
        self,
        provider: BaseWebSearchProvider,
        cache_dir: str | Path = ".cache/web_search",
        ttl_seconds: int = 86400,
    ) -> None:
        self._provider = provider
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ttl_seconds = ttl_seconds
        logger.debug(
            "Initialized CachedProvider cache at %s with TTL=%ss",
            self._cache_dir.absolute(),
            self._ttl_seconds,
        )

    async def validate(self) -> None:
        await self._provider.validate()

    # ---------- Cache utils ----------

    def _generate_cache_key(
        self,
        query: str,
        num_links: int,
        offset: int,
        result_filter: str,
        safesearch: bool,
        freshness: str | None,
        spellcheck: bool,
        language: str | None,
    ) -> str:
        params = {
            "query": query.strip().lower(),
            "num_links": num_links,
            "offset": offset,
            "result_filter": result_filter,
            "safesearch": safesearch,
            "freshness": freshness,
            "spellcheck": spellcheck,
            "language": language,
        }
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(params_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        return self._cache_dir / f"{cache_key}.json"

    def _is_expired(self, created_ts: float) -> bool:
        return (time.time() - created_ts) > self._ttl_seconds

    def _load_from_cache(self, cache_key: str) -> list[WebSearchResult] | None:
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, encoding="utf-8") as f:
                payload = json.load(f)

            created_at = payload.get("created_at")
            if created_at is None or self._is_expired(created_at):
                try:
                    cache_path.unlink(missing_ok=True)
                except OSError:
                    logger.debug("Failed to delete expired cache file %s", cache_path)
                logger.debug("Cache expired for %s", cache_path.name)
                return None

            items = payload.get("results", [])
            results = [WebSearchResult(**item) for item in items]
            logger.debug(
                "Cache hit: loaded %d results from %s",
                len(results),
                cache_path.name,
            )
            return results
        except Exception as e:
            logger.warning("Failed to load cache from %s: %s", cache_path, e)
            return None

    def _save_to_cache(self, cache_key: str, results: list[WebSearchResult]) -> None:
        cache_path = self._get_cache_path(cache_key)
        try:
            data = [
                {
                    "idx": r.idx,
                    "title": r.title,
                    "url": r.url,
                    "description": r.description,
                    "age": r.age,
                }
                for r in results
            ]
            payload = {
                "created_at": time.time(),
                "results": data,
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.debug("Cached %d results to %s", len(results), cache_path.name)
        except Exception as e:
            logger.warning("Failed to save cache to %s: %s", cache_path, e)

    async def make_query(
        self,
        query: str,
        num_links: int = 10,
        offset: int = 0,
        result_filter: str = "web",
        safesearch: bool = True,
        freshness: str | None = None,
        spellcheck: bool = True,
        language: str | None = None,
        **kwargs: Any,
    ) -> list[WebSearchResult]:
        cache_key = await asyncio.to_thread(
            self._generate_cache_key,
            query=query,
            num_links=num_links,
            offset=offset,
            result_filter=result_filter,
            safesearch=safesearch,
            freshness=freshness,
            spellcheck=spellcheck,
            language=language,
        )

        cached_results = await asyncio.to_thread(self._load_from_cache, cache_key)
        if cached_results is not None:
            return cached_results

        logger.debug("Cache miss for query '%s' - calling provider", query)
        results = await self._provider.make_query(
            query=query,
            num_links=num_links,
            offset=offset,
            result_filter=result_filter,
            safesearch=safesearch,
            freshness=freshness,
            spellcheck=spellcheck,
            language=language,
            **kwargs,
        )

        await asyncio.to_thread(self._save_to_cache, cache_key, results)
        return results

    async def close(self) -> None:
        await self._provider.close()
