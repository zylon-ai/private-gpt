import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.components.web.web_search.providers.brave import BraveSearchProvider


# With mocks
@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.brave.api_key = "api-key"
    settings.brave.rate_limit = 1.0
    settings.brave.timeout = 30
    settings.web_search.enabled = True
    return settings


@pytest.fixture
def provider(mock_settings):
    return BraveSearchProvider(mock_settings)


@pytest.fixture
def mock_brave_response():
    return {
        "web": {
            "results": [
                {
                    "title": "Result 1",
                    "url": "https://example.com/1",
                    "description": "Description 1",
                    "page_age": "2024-01-01",
                },
                {
                    "title": "Result 2",
                    "url": "https://example.com/2",
                    "description": "Description 2",
                    "page_age": "2024-01-02",
                },
            ]
        }
    }


class TestRateLimiting:
    """Tests to verify rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_respect_rate_limit(
        self, provider, mock_brave_response
    ):
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(0.1)
            return mock_brave_response

        provider._execute_http_request = AsyncMock(side_effect=slow_response)

        # Measure time
        start = asyncio.get_event_loop().time()

        # Launch 5 concurrent requests
        queries = [f"query_{i}" for i in range(5)]
        tasks = [provider.make_query(q) for q in queries]
        results = await asyncio.gather(*tasks)

        elapsed = asyncio.get_event_loop().time() - start

        # Assertions
        assert len(results) == 5
        assert all(len(r) == 2 for r in results)

        # With rate_limit=1.0 (1 req/s), 5 requests need 4 intervals
        # Minimum: 4.0 seconds
        assert elapsed >= 4.0, (
            f"Rate limiting failed: elapsed={elapsed:.2f}s (expected >=4.0s)"
        )
        assert elapsed < 6.0, f"Too slow: elapsed={elapsed:.2f}s (expected <5.0s)"

        # Verify it was called 5 times
        assert provider._execute_http_request.call_count == 5

    @pytest.mark.asyncio
    async def test_different_rate_limits(self, mock_settings, mock_brave_response):
        """Test with different rate limiting values."""
        # Rate limit of 2 requests/second
        mock_settings.brave.rate_limit = 2.0
        provider = BraveSearchProvider(mock_settings)

        provider._execute_http_request = AsyncMock(return_value=mock_brave_response)

        start = asyncio.get_event_loop().time()

        # 4 requests with rate=2.0 should take ~1.5 seconds
        tasks = [provider.make_query(f"query_{i}") for i in range(4)]
        await asyncio.gather(*tasks)

        elapsed = asyncio.get_event_loop().time() - start

        # With rate_limit=2.0, minimum interval is 0.5s
        # 4 requests = 3 intervals = 1.5s minimum
        assert elapsed >= 1.5, f"Rate limiting failed: elapsed={elapsed:.2f}s"
        assert elapsed < 3, f"Too slow: elapsed={elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_rate_limiting_with_slow_api_responses(
        self, provider, mock_brave_response
    ):
        """Test that rate limiting works even when API responses are slow."""
        call_times = []

        async def slow_api(*args, **kwargs):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.5)  # Slow API (500ms)
            return mock_brave_response

        provider._execute_http_request = AsyncMock(side_effect=slow_api)

        start = asyncio.get_event_loop().time()

        # 3 concurrent requests
        tasks = [provider.make_query(f"query_{i}") for i in range(3)]
        await asyncio.gather(*tasks)

        elapsed = asyncio.get_event_loop().time() - start

        # Verify that intervals between calls respect rate limit
        for i in range(1, len(call_times)):
            interval = call_times[i] - call_times[i - 1]
            assert interval >= 0.99, f"Interval {i} too short: {interval:.3f}s"

        # Total: 2 intervals of 1s + 3 calls of 0.5s = 2s + 0.5s = 2.5s
        # (calls overlap with intervals)
        assert elapsed >= 2.0, f"Too fast: elapsed={elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_multiple_users_concurrent_requests(
        self, provider, mock_brave_response
    ):
        """Test simulating multiple users making simultaneous requests."""
        request_timestamps = []

        async def track_timestamp(*args, **kwargs):
            request_timestamps.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.01)
            return mock_brave_response

        provider._execute_http_request = AsyncMock(side_effect=track_timestamp)

        # Simulate 3 users, each makes 2 requests
        async def user_requests(user_id: int):
            results = []
            for i in range(2):
                result = await provider.make_query(f"user_{user_id}_query_{i}")
                results.append(result)
            return results

        start = asyncio.get_event_loop().time()

        # 3 concurrent users
        user_tasks = [user_requests(user_id) for user_id in range(3)]
        all_results = await asyncio.gather(*user_tasks)

        elapsed = asyncio.get_event_loop().time() - start

        # Verify all requests were processed
        assert len(all_results) == 3
        assert all(len(user_results) == 2 for user_results in all_results)

        # 6 total requests = 5 intervals of 1s
        assert elapsed >= 5.0, f"Global rate limiting failed: elapsed={elapsed:.2f}s"

        # Verify timestamps - each request should be separated by ~1s
        for i in range(1, len(request_timestamps)):
            interval = request_timestamps[i] - request_timestamps[i - 1]
            assert interval >= 0.99, f"Interval {i} too short: {interval:.3f}s"

    @pytest.mark.asyncio
    async def test_single_request_no_delay(self, provider, mock_brave_response):
        """Test that a single request has no delay."""
        provider._execute_http_request = AsyncMock(return_value=mock_brave_response)

        start = asyncio.get_event_loop().time()
        await provider.make_query("single_query")
        elapsed = asyncio.get_event_loop().time() - start

        # Single request should be almost instantaneous (<100ms)
        assert elapsed < 0.1, f"Single request too slow: {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_sequential_requests_respect_rate_limit(
        self, provider, mock_brave_response
    ):
        """Test that sequential (non-concurrent) requests also respect rate limit."""
        provider._execute_http_request = AsyncMock(return_value=mock_brave_response)

        start = asyncio.get_event_loop().time()

        # Make requests sequentially (DO NOT use gather)
        await provider.make_query("query_1")
        await provider.make_query("query_2")
        await provider.make_query("query_3")

        elapsed = asyncio.get_event_loop().time() - start

        # 3 requests = 2 intervals = 2s minimum
        assert elapsed >= 2.0, f"Rate limiting failed: elapsed={elapsed:.2f}s"
