from pathlib import Path
from unittest.mock import patch

import pytest

from private_gpt.artifact_index.base_artifact_index import (
    ArtifactIndexStatus,
    IndexNotReadyException,
)
from private_gpt.server.ingest.ingest_service import IngestService
from tests.fixtures.mock_injector import MockInjector


def test_populate_non_existent_vector_index_fails(injector: MockInjector) -> None:
    service: IngestService = injector.get(IngestService)

    with patch(
        "private_gpt.artifact_index.vector_artifact_index.VectorArtifactIndex.status",
        return_value=ArtifactIndexStatus.NOT_INITIALIZED,
    ):
        with pytest.raises(ValueError) as error:
            service.populate_vector_index(
                collection="test_collection",
                artifact="non_existent_artifact",
                file_data=Path("test"),
            )
        assert error is not None


def test_delete_non_populated_vector_index_fails(injector: MockInjector) -> None:
    service: IngestService = injector.get(IngestService)

    with patch(
        "private_gpt.artifact_index.vector_artifact_index.VectorArtifactIndex.status",
        return_value=ArtifactIndexStatus.NOT_INITIALIZED,
    ):
        with pytest.raises(ValueError) as error:
            service.delete(
                collection="test_collection",
                artifact="non_existent_artifact",
            )
        assert error is not None

    with patch(
        "private_gpt.artifact_index.vector_artifact_index.VectorArtifactIndex.status",
        return_value=ArtifactIndexStatus.INITIALIZED,
    ):
        with pytest.raises(IndexNotReadyException) as error:
            service.delete(
                collection="test_collection",
                artifact="non_existent_artifact",
            )
        assert error is not None
