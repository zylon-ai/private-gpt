import io
from urllib.parse import quote

import anthropic
from fastapi.testclient import TestClient

from tests.fixtures.anthropic_httpx2_client import create_mock_http_client
from tests.server.files.conftest import (
    _FILE_CONTENT,
    _FILE_NAME,
    _MIME_TYPE,
    _SESSION_ID,
)

_CLIENT_KWARGS = {
    "base_url": "http://testserver",
    "api_key": "test_key",
    "max_retries": 0,
}


def _file_url(file_id: str, scope_id: str, suffix: str = "") -> str:
    encoded = quote(file_id, safe="")
    return f"/v1/files/{encoded}{suffix}?scope_id={scope_id}"


def _sdk_client(test_client: TestClient) -> anthropic.Anthropic:
    """Return an Anthropic SDK client whose HTTP layer is bridged to *test_client*.

    Anthropic 1.x moved its HTTP layer from ``httpx`` to ``httpx2``, so the
    client's internal transport must be an ``httpx2`` client rather than an
    ``httpx`` one (and cannot be driven via pytest-httpx, which only patches
    ``httpx``). See ``tests/fixtures/anthropic_httpx2_client.py``.
    """
    client = anthropic.Anthropic(**_CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client)
    return client


def test_sdk_upload_parses_as_file_metadata(files_client: TestClient) -> None:
    client = _sdk_client(files_client)

    result = client.beta.files.upload(
        file=(_FILE_NAME, io.BytesIO(_FILE_CONTENT), _MIME_TYPE),
        extra_query={"scope_id": _SESSION_ID},
    )

    assert isinstance(result, anthropic.types.beta.BetaFileMetadata)
    assert result.filename == _FILE_NAME
    assert result.size_bytes == len(_FILE_CONTENT)
    assert result.type == "file"


def test_sdk_list_parses_as_sync_page(files_client: TestClient) -> None:
    files_client.post(
        f"/v1/files?scope_id={_SESSION_ID}",
        files={"file": (_FILE_NAME, io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
    )

    client = _sdk_client(files_client)
    page = client.beta.files.list(scope_id=_SESSION_ID)

    items = list(page)
    assert len(items) == 1
    assert isinstance(items[0], anthropic.types.beta.BetaFileMetadata)
    assert items[0].filename == _FILE_NAME


def test_sdk_retrieve_metadata(files_client: TestClient) -> None:
    upload_resp = files_client.post(
        f"/v1/files?scope_id={_SESSION_ID}",
        files={"file": (_FILE_NAME, io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
    )
    file_id = upload_resp.json()["id"]

    client = _sdk_client(files_client)
    meta = client.beta.files.retrieve_metadata(
        file_id, extra_query={"scope_id": _SESSION_ID}
    )

    assert isinstance(meta, anthropic.types.beta.BetaFileMetadata)
    assert meta.id == file_id


def test_sdk_download_returns_bytes(files_client: TestClient) -> None:
    upload_resp = files_client.post(
        f"/v1/files?scope_id={_SESSION_ID}",
        files={"file": (_FILE_NAME, io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
    )
    file_id = upload_resp.json()["id"]

    client = _sdk_client(files_client)
    response = client.beta.files.download(
        file_id, extra_query={"scope_id": _SESSION_ID}
    )

    assert response.read() == _FILE_CONTENT


def test_sdk_delete_parses_as_deleted_file(files_client: TestClient) -> None:
    upload_resp = files_client.post(
        f"/v1/files?scope_id={_SESSION_ID}",
        files={"file": (_FILE_NAME, io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
    )
    file_id = upload_resp.json()["id"]

    client = _sdk_client(files_client)
    deleted = client.beta.files.delete(file_id, extra_query={"scope_id": _SESSION_ID})

    assert isinstance(deleted, anthropic.types.beta.BetaDeletedFile)
    assert deleted.id == file_id
    assert deleted.type == "file_deleted"
