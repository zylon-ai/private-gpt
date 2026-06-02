import contextlib

import pytest

from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from tests.fixtures.mock_injector import MockInjector


@pytest.fixture(autouse=True)
def _auto_close_vector_store_client(injector: MockInjector) -> None:
    """Auto close VectorStore client after each test.

    VectorStore client opens a connection to the database that causes
    issues when running tests too fast, so close explicitly after each
    test.
    """
    yield
    with contextlib.suppress(Exception):
        # Avoid to close the client if it's already closed
        injector.get(VectorStoreComponent).close()
