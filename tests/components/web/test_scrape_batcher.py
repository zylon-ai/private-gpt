import asyncio

import pytest

from private_gpt.components.web.scraper.batcher import ScrapeBatcher


class _Recorder:
    def __init__(self, results: dict[str, str | Exception] | None = None) -> None:
        self.batches: list[list[str]] = []
        self.results = results or {}

    async def run_batch(
        self, urls: list[str], timeout_seconds: int
    ) -> list[str | Exception]:
        self.batches.append(urls)
        return [self.results.get(url, f"<html>{url}</html>") for url in urls]


async def test_concurrent_requests_share_one_batch() -> None:
    recorder = _Recorder()
    batcher = ScrapeBatcher(
        batch_size=5, batch_wait_seconds=0.05, run_batch=recorder.run_batch
    )

    results = await asyncio.gather(
        batcher.submit("https://a.com", 10),
        batcher.submit("https://b.com", 10),
        batcher.submit("https://c.com", 10),
    )

    assert results == [
        "<html>https://a.com</html>",
        "<html>https://b.com</html>",
        "<html>https://c.com</html>",
    ]
    # one browser run for the three pages, not three
    assert recorder.batches == [["https://a.com", "https://b.com", "https://c.com"]]


async def test_full_batch_dispatches_before_window_closes() -> None:
    recorder = _Recorder()
    batcher = ScrapeBatcher(
        batch_size=2, batch_wait_seconds=60, run_batch=recorder.run_batch
    )

    results = await asyncio.wait_for(
        asyncio.gather(
            batcher.submit("https://a.com", 10),
            batcher.submit("https://b.com", 10),
        ),
        timeout=1,
    )

    assert len(results) == 2
    assert recorder.batches == [["https://a.com", "https://b.com"]]


async def test_overflow_starts_a_new_batch() -> None:
    recorder = _Recorder()
    batcher = ScrapeBatcher(
        batch_size=2, batch_wait_seconds=0.05, run_batch=recorder.run_batch
    )

    await asyncio.gather(
        batcher.submit("https://a.com", 10),
        batcher.submit("https://b.com", 10),
        batcher.submit("https://c.com", 10),
    )

    assert [len(b) for b in recorder.batches] == [2, 1]


async def test_batch_size_one_dispatches_immediately() -> None:
    recorder = _Recorder()
    batcher = ScrapeBatcher(
        batch_size=1, batch_wait_seconds=60, run_batch=recorder.run_batch
    )

    result = await asyncio.wait_for(batcher.submit("https://a.com", 10), timeout=1)

    assert result == "<html>https://a.com</html>"
    assert recorder.batches == [["https://a.com"]]


async def test_per_request_errors_only_fail_their_future() -> None:
    boom = RuntimeError("boom")
    recorder = _Recorder(results={"https://bad.com": boom})
    batcher = ScrapeBatcher(
        batch_size=2, batch_wait_seconds=0.05, run_batch=recorder.run_batch
    )

    results = await asyncio.gather(
        batcher.submit("https://ok.com", 10),
        batcher.submit("https://bad.com", 10),
        return_exceptions=True,
    )

    assert results[0] == "<html>https://ok.com</html>"
    assert results[1] is boom


async def test_run_level_failure_fails_the_whole_batch() -> None:
    async def run_batch(urls, timeout_seconds):
        raise RuntimeError("session exploded")

    batcher = ScrapeBatcher(batch_size=2, batch_wait_seconds=0.05, run_batch=run_batch)

    results = await asyncio.gather(
        batcher.submit("https://a.com", 10),
        batcher.submit("https://b.com", 10),
        return_exceptions=True,
    )

    assert all(isinstance(r, RuntimeError) for r in results)


async def test_close_fails_pending_requests() -> None:
    recorder = _Recorder()
    batcher = ScrapeBatcher(
        batch_size=5, batch_wait_seconds=60, run_batch=recorder.run_batch
    )

    pending = asyncio.create_task(batcher.submit("https://a.com", 10))
    await asyncio.sleep(0.01)
    await batcher.close()

    with pytest.raises(RuntimeError, match="closed"):
        await pending
    assert recorder.batches == []
