import io
from typing import Any
from urllib.parse import quote

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock

from tests.server.files.conftest import (
    _FILE_CONTENT,
    _FILE_NAME,
    _MIME_TYPE,
    _SESSION_ID,
)

_CLIENT_KWARGS: dict[str, Any] = {
    "base_url": "http://testserver",
    "api_key": "test_key",
    "max_retries": 0,
}


def _file_url(file_id: str, scope_id: str, suffix: str = "") -> str:
    encoded = quote(file_id, safe="")
    return f"/v1/files/{encoded}{suffix}?scope_id={scope_id}"


def _sdk_client(
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> anthropic.Anthropic:
    """Return an Anthropic SDK client whose HTTP layer is bridged to *test_client*."""

    def _build(request: httpx.Request) -> httpx.Response:
        resp = test_client.send(
            test_client.build_request(
                method=request.method,
                url=request.url.path,
                headers=request.headers,
                content=request.content,
                params=request.url.params,
            )
        )
        ct = resp.headers.get("Content-Type", "")
        if "application/json" in ct:
            return httpx.Response(
                status_code=resp.status_code,
                headers=resp.headers,
                json=resp.json(),
            )
        return httpx.Response(
            status_code=resp.status_code,
            headers=resp.headers,
            content=resp.content,
        )

    httpx_mock.add_callback(_build)
    httpx_mock.add_response(is_reusable=True)

    client = anthropic.Anthropic(**_CLIENT_KWARGS)
    client._client = httpx.Client()
    return client


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_sdk_upload_parses_as_file_metadata(
    files_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    client = _sdk_client(files_client, httpx_mock)

    result = client.beta.files.upload(
        file=(_FILE_NAME, io.BytesIO(_FILE_CONTENT), _MIME_TYPE),
        extra_query={"scope_id": _SESSION_ID},
    )

    assert isinstance(result, anthropic.types.beta.FileMetadata)
    assert result.filename == _FILE_NAME
    assert result.size_bytes == len(_FILE_CONTENT)
    assert result.type == "file"


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_sdk_list_parses_as_sync_page(
    files_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    files_client.post(
        f"/v1/files?scope_id={_SESSION_ID}",
        files={"file": (_FILE_NAME, io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
    )

    client = _sdk_client(files_client, httpx_mock)
    page = client.beta.files.list(scope_id=_SESSION_ID)

    items = list(page)
    assert len(items) == 1
    assert isinstance(items[0], anthropic.types.beta.FileMetadata)
    assert items[0].filename == _FILE_NAME


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_sdk_retrieve_metadata(
    files_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    upload_resp = files_client.post(
        f"/v1/files?scope_id={_SESSION_ID}",
        files={"file": (_FILE_NAME, io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
    )
    file_id = upload_resp.json()["id"]

    client = _sdk_client(files_client, httpx_mock)
    meta = client.beta.files.retrieve_metadata(
        file_id, extra_query={"scope_id": _SESSION_ID}
    )

    assert isinstance(meta, anthropic.types.beta.FileMetadata)
    assert meta.id == file_id


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_sdk_download_returns_bytes(
    files_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    upload_resp = files_client.post(
        f"/v1/files?scope_id={_SESSION_ID}",
        files={"file": (_FILE_NAME, io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
    )
    file_id = upload_resp.json()["id"]

    client = _sdk_client(files_client, httpx_mock)
    response = client.beta.files.download(
        file_id, extra_query={"scope_id": _SESSION_ID}
    )

    assert response.read() == _FILE_CONTENT


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_sdk_delete_parses_as_deleted_file(
    files_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    upload_resp = files_client.post(
        f"/v1/files?scope_id={_SESSION_ID}",
        files={"file": (_FILE_NAME, io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
    )
    file_id = upload_resp.json()["id"]

    client = _sdk_client(files_client, httpx_mock)
    deleted = client.beta.files.delete(file_id, extra_query={"scope_id": _SESSION_ID})

    assert isinstance(deleted, anthropic.types.beta.DeletedFile)
    assert deleted.id == file_id
    assert deleted.type == "file_deleted"
