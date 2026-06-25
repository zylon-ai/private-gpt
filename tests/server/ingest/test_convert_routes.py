import base64
from pathlib import Path

from fastapi.testclient import TestClient

from private_gpt.server.ingest.convert_router import (
    ConvertBody,
    ConvertResponse,
    ReadersResponse,
)
from private_gpt.server.utils.artifact_input import FileArtifact, TextArtifact


def _encode_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def test_list_readers_returns_known_readers(test_client: TestClient) -> None:
    response = test_client.get("/v1/artifacts/readers")
    assert response.status_code == 200
    result = ReadersResponse.model_validate(response.json())
    assert len(result.data) > 0
    # At minimum text reader should always be present
    known = {"markitdown", "docling", "text", "pptx2md"}
    assert known & set(result.data.keys()), "Expected at least one known reader"


def test_list_readers_response_has_extensions(test_client: TestClient) -> None:
    response = test_client.get("/v1/artifacts/readers")
    result = ReadersResponse.model_validate(response.json())
    for reader_name, info in result.data.items():
        assert len(info.extensions) > 0, f"Reader '{reader_name}' has no extensions"
        for ext in info.extensions:
            assert ext.startswith("."), f"Extension '{ext}' should start with '.'"


def test_convert_plain_text_returns_markdown(test_client: TestClient) -> None:
    body = ConvertBody(
        input=TextArtifact(value="Hello world"),
        metadata={"file_name": "hello.txt"},
    )
    response = test_client.post(
        "/v1/artifacts/convert",
        json=body.model_dump(exclude_none=True),
    )
    assert response.status_code == 200
    result = ConvertResponse.model_validate(response.json())
    assert isinstance(result.content, str)
    assert "Hello world" in result.content
    assert result.reader


def test_convert_txt_file_returns_markdown(test_client: TestClient) -> None:
    path = Path(__file__).parent / "test.txt"
    body = ConvertBody(
        input=FileArtifact(value=_encode_file(path)),
        metadata={"file_name": "test.txt"},
    )
    response = test_client.post(
        "/v1/artifacts/convert",
        json=body.model_dump(exclude_none=True),
    )
    assert response.status_code == 200
    result = ConvertResponse.model_validate(response.json())
    assert isinstance(result.content, str)
    assert len(result.content) > 0
    assert result.reader == "text"


def test_convert_with_explicit_reader(test_client: TestClient) -> None:
    path = Path(__file__).parent / "test.txt"
    body = ConvertBody(
        input=FileArtifact(value=_encode_file(path)),
        metadata={"file_name": "test.txt"},
        reader="text",
    )
    response = test_client.post(
        "/v1/artifacts/convert",
        json=body.model_dump(exclude_none=True),
    )
    assert response.status_code == 200
    result = ConvertResponse.model_validate(response.json())
    assert result.reader == "text"


def test_convert_rejects_invalid_reader_for_extension(test_client: TestClient) -> None:
    path = Path(__file__).parent / "test.txt"
    body = ConvertBody(
        input=FileArtifact(value=_encode_file(path)),
        metadata={"file_name": "test.txt"},
        reader="docling",  # docling does not support .txt
    )
    response = test_client.post(
        "/v1/artifacts/convert",
        json=body.model_dump(exclude_none=True),
    )
    assert response.status_code == 422


def test_convert_object_format_returns_tree(test_client: TestClient) -> None:
    body = ConvertBody(
        input=TextArtifact(value="# Title\n\nSome content here."),
        metadata={"file_name": "doc.md"},
        format="object",
    )
    response = test_client.post(
        "/v1/artifacts/convert",
        json=body.model_dump(exclude_none=True),
    )
    assert response.status_code == 200
    result = response.json()
    # object format returns a dict (ContentTree), not a plain string
    assert isinstance(result["content"], dict)
    assert "children" in result["content"]
