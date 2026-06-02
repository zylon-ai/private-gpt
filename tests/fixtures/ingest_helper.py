import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from private_gpt.server.ingest.ingest_router import (
    DeleteIngestedDocumentBody,
    IngestBody,
    IngestResponse,
)
from private_gpt.server.utils.artifact_input import FileArtifact


class IngestHelper:
    def __init__(self, test_client: TestClient):
        self.test_client = test_client

    def ingest_file(
        self, path: Path, artifact: str | None = None, collection: str = "default"
    ) -> IngestResponse:
        file_content = path.read_bytes()
        base64_content = base64.b64encode(file_content).decode("utf-8")

        filename = path.name
        artifact = artifact or filename

        body = IngestBody(
            input=FileArtifact(value=base64_content),
            artifact=artifact,
            collection=collection,
            metadata={"file_name": filename},
        )

        response = self.test_client.post(
            "/v1/artifacts/ingest",
            json=body.model_dump(exclude_none=True, by_alias=True),
        )
        assert response.status_code == 200
        ingest_result = IngestResponse.model_validate(response.json())
        return ingest_result

    def delete_file(self, collection: str, artifact: str) -> None:
        body = DeleteIngestedDocumentBody(collection=collection, artifact=artifact)
        response = self.test_client.post("/v1/artifacts/delete", json=body.model_dump())
        assert response.status_code == 200


@pytest.fixture
def ingest_helper(test_client: TestClient) -> IngestHelper:
    return IngestHelper(test_client)
