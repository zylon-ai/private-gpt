import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest
from qdrant_client import QdrantClient, models

from private_gpt.components.vector_store.qdrant_client_builder import (
    QdrantClientBuilder,
)

_COLLECTION = "test"
_BATCH = 50
_BATCHES = 100


def _local_client() -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection(
        _COLLECTION,
        vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
    )
    return client


def _upload(client: QdrantClient, start: int) -> None:
    client.upload_points(
        collection_name=_COLLECTION,
        points=[
            models.PointStruct(id=i, vector=[0.1, 0.2, 0.3, 0.4], payload={"g": "a"})
            for i in range(start, start + _BATCH)
        ],
        batch_size=_BATCH,
        parallel=1,
        wait=True,
    )


def test_local_client_survives_parallel_uploads_and_scrolls() -> None:
    # Mirrors PatchedQdrantVectorStore: add() uploads batches from a thread pool
    # while get_nodes() scrolls with a payload filter from its producer thread.
    client = _local_client()
    QdrantClientBuilder.serialize_local_access(client)
    scroll_filter = models.Filter(
        must=[models.FieldCondition(key="g", match=models.MatchValue(value="a"))]
    )
    errors: list[BaseException] = []
    done = threading.Event()

    def scroll() -> None:
        while not done.is_set():
            try:
                client.scroll(
                    _COLLECTION,
                    scroll_filter=scroll_filter,
                    limit=100,
                    with_payload=True,
                )
            except BaseException as exc:
                errors.append(exc)
                return
            time.sleep(0)

    reader = threading.Thread(target=scroll)
    reader.start()
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(
                pool.map(
                    lambda start: _upload(client, start),
                    range(0, _BATCH * _BATCHES, _BATCH),
                )
            )
        assert errors == []
        assert client.count(_COLLECTION).count == _BATCH * _BATCHES
    finally:
        done.set()
        reader.join()
        client.close()


def test_remote_client_is_left_unchanged() -> None:
    client = QdrantClient(url="http://localhost:6333")
    remote = client._client
    before = dict(vars(remote))

    QdrantClientBuilder.serialize_local_access(client)

    assert dict(vars(remote)) == before
    client.close()


def test_build_clients_serializes_local_path_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only the sync client's wiring is under test here.
    monkeypatch.setattr(
        QdrantClientBuilder, "clean_lock", staticmethod(lambda db_dir: None)
    )
    monkeypatch.setattr(
        "private_gpt.components.vector_store.qdrant_client_builder.AsyncQdrantClient",
        Mock(),
    )
    settings = Mock()
    settings.qdrant.get_parameters.return_value = {"path": str(tmp_path / "qdrant")}

    clients = QdrantClientBuilder.build_clients(settings)
    try:
        local = clients.client._client
        # Wrapped methods are bound per instance; the class attribute is untouched.
        assert "upsert" in vars(local)
        assert "scroll" in vars(local)
        assert local.scroll.__wrapped__.__func__ is type(local).scroll
    finally:
        clients.close()
