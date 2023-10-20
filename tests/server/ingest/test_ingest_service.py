from unittest.mock import PropertyMock, patch

from llama_index import Document

from private_gpt.server.ingest.ingest_service import IngestService
from tests.fixtures.mock_injector import MockInjector


def test_save_many_nodes(injector: MockInjector) -> None:
    """This is a specific test for a local Chromadb Vector Database setup.

    Extend it when we add support for other vector databases in VectorStoreComponent.
    """
    with patch(
        "chromadb.api.segment.SegmentAPI.max_batch_size", new_callable=PropertyMock
    ) as max_batch_size:
        # Make max batch size of Chromadb very small
        max_batch_size.return_value = 10

        ingest_service = injector.get(IngestService)

        documents = []
        for _i in range(100):
            documents.append(Document(text="This is a sentence."))

        ingested_docs = ingest_service._save_docs(documents)
        assert len(ingested_docs) == len(documents)
