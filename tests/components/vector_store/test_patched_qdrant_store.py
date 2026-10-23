from unittest.mock import AsyncMock, Mock, call

import pytest

from private_gpt.components.vector_store.patched_qdrant_store import (
    PatchedQdrantVectorStore,
)


class _Executor:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def map(self, function: object, batches: list[list[str]]) -> list[None]:
        results = []
        for batch in batches:
            self.batches.append(batch)
            results.append(function(batch))  # type: ignore[operator]
        return results


def test_add_uses_executor_with_fork_free_qdrant_uploads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = PatchedQdrantVectorStore.model_construct()
    client = Mock()
    executor = _Executor()
    monkeypatch.setattr(
        PatchedQdrantVectorStore,
        "executor",
        classmethod(lambda cls, **kwargs: executor),
    )
    object.__setattr__(store, "_collection_initialized", True)
    object.__setattr__(store, "_legacy_vector_format", True)
    object.__setattr__(store, "_client", client)
    object.__setattr__(store, "collection_name", "test")
    object.__setattr__(store, "sparse_vector_name", "sparse")
    object.__setattr__(store, "batch_size", 2)
    object.__setattr__(store, "max_retries", 1)
    object.__setattr__(store, "parallel", 3)
    object.__setattr__(
        store,
        "_build_points",
        Mock(return_value=(["p1", "p2", "p3", "p4", "p5"], ["id"])),
    )

    assert store.add([Mock()]) == ["id"]

    assert executor.batches == [["p1", "p2"], ["p3", "p4"], ["p5"]]
    assert client.upload_points.call_args_list == [
        call(
            collection_name="test",
            points=batch,
            batch_size=2,
            parallel=1,
            max_retries=1,
            wait=True,
            shard_key_selector=None,
        )
        for batch in executor.batches
    ]


async def test_async_add_uses_bounded_batch_consumers() -> None:
    store = PatchedQdrantVectorStore.model_construct()
    async_client = AsyncMock()
    object.__setattr__(store, "_client", Mock(_client=object()))
    object.__setattr__(store, "_aclient", async_client)
    object.__setattr__(store, "_collection_initialized", True)
    object.__setattr__(store, "_legacy_vector_format", True)
    object.__setattr__(store, "collection_name", "test")
    object.__setattr__(store, "sparse_vector_name", "sparse")
    object.__setattr__(store, "batch_size", 2)
    object.__setattr__(store, "max_retries", 1)
    object.__setattr__(store, "parallel", 2)
    object.__setattr__(store, "_ensure_async_client", Mock())
    object.__setattr__(store, "_acollection_exists", AsyncMock(return_value=True))
    object.__setattr__(
        store,
        "_build_points",
        Mock(return_value=(["p1", "p2", "p3", "p4", "p5"], ["id"])),
    )

    assert await store.async_add([Mock()]) == ["id"]

    async_client.upsert.assert_has_awaits(
        [
            call(
                collection_name="test",
                points=["p1", "p2"],
                wait=True,
                shard_key_selector=None,
            ),
            call(
                collection_name="test",
                points=["p3", "p4"],
                wait=True,
                shard_key_selector=None,
            ),
            call(
                collection_name="test",
                points=["p5"],
                wait=True,
                shard_key_selector=None,
            ),
        ],
        any_order=True,
    )
