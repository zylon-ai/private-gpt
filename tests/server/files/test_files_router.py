"""Direct router tests for POST/GET/DELETE /v1/files."""

import io
from pathlib import Path
from urllib.parse import quote

from fastapi.testclient import TestClient

from private_gpt.server.files.file_models import (
    DeletedFile,
    FileListResponse,
    FileMetadata,
)
from tests.server.files.conftest import (
    _FILE_CONTENT,
    _FILE_NAME,
    _MIME_TYPE,
    _SESSION_ID,
)


def _file_url(file_id: str, scope_id: str, suffix: str = "") -> str:
    """Build a URL for a file endpoint, URL-encoding the absolute path ID."""
    encoded = quote(file_id, safe="")
    return f"/v1/files/{encoded}{suffix}?scope_id={scope_id}"


def test_upload_returns_file_metadata(
    files_client: TestClient, volume_root: Path
) -> None:
    resp = files_client.post(
        f"/v1/files?scope_id={_SESSION_ID}",
        files={"file": (_FILE_NAME, io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
    )
    assert resp.status_code == 200
    meta = FileMetadata.model_validate(resp.json())
    assert meta.filename == _FILE_NAME
    assert meta.size_bytes == len(_FILE_CONTENT)
    assert meta.downloadable is False
    assert meta.scope.id == _SESSION_ID
    # ID must be an absolute path pointing into uploads/
    assert meta.id.startswith("/")
    assert "uploads" in meta.id
    assert meta.id.endswith(_FILE_NAME)


def test_list_files_empty_session(files_client: TestClient) -> None:
    resp = files_client.get(f"/v1/files?scope_id={_SESSION_ID}")
    assert resp.status_code == 200
    listing = FileListResponse.model_validate(resp.json())
    assert listing.data == []
    assert listing.has_more is False


def test_list_files_after_upload(files_client: TestClient) -> None:
    files_client.post(
        f"/v1/files?scope_id={_SESSION_ID}",
        files={"file": (_FILE_NAME, io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
    )

    resp = files_client.get(f"/v1/files?scope_id={_SESSION_ID}")
    assert resp.status_code == 200
    listing = FileListResponse.model_validate(resp.json())
    assert len(listing.data) == 1
    assert listing.data[0].filename == _FILE_NAME


def test_get_file_metadata(files_client: TestClient) -> None:
    upload_resp = files_client.post(
        f"/v1/files?scope_id={_SESSION_ID}",
        files={"file": (_FILE_NAME, io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
    )
    file_id = upload_resp.json()["id"]

    resp = files_client.get(_file_url(file_id, _SESSION_ID))
    assert resp.status_code == 200
    meta = FileMetadata.model_validate(resp.json())
    assert meta.id == file_id
    assert meta.filename == _FILE_NAME


def test_get_file_metadata_not_found(files_client: TestClient) -> None:
    resp = files_client.get(_file_url("/nonexistent/path/file.txt", _SESSION_ID))
    assert resp.status_code == 404


def test_download_uploaded_file(files_client: TestClient) -> None:
    upload_resp = files_client.post(
        f"/v1/files?scope_id={_SESSION_ID}",
        files={"file": (_FILE_NAME, io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
    )
    file_id = upload_resp.json()["id"]

    resp = files_client.get(_file_url(file_id, _SESSION_ID, suffix="/content"))
    assert resp.status_code == 200
    assert resp.content == _FILE_CONTENT


def test_download_output_file(files_client: TestClient, volume_root: Path) -> None:
    """A sandbox output file can be downloaded via its absolute path ID."""
    output_path = volume_root / "sessions" / _SESSION_ID / "outputs" / "result.csv"
    output_path.write_bytes(_FILE_CONTENT)
    file_id = str(output_path.resolve())

    resp = files_client.get(_file_url(file_id, _SESSION_ID, suffix="/content"))
    assert resp.status_code == 200
    assert resp.content == _FILE_CONTENT


def test_list_includes_outputs(files_client: TestClient, volume_root: Path) -> None:
    output_path = volume_root / "sessions" / _SESSION_ID / "outputs" / "result.png"
    output_path.write_bytes(b"\x89PNG")
    output_id = str(output_path.resolve())

    resp = files_client.get(f"/v1/files?scope_id={_SESSION_ID}")
    listing = FileListResponse.model_validate(resp.json())
    ids = [f.id for f in listing.data]
    assert output_id in ids
    downloadable = {f.id: f.downloadable for f in listing.data}
    assert downloadable[output_id] is True


def test_delete_uploaded_file(files_client: TestClient) -> None:
    upload_resp = files_client.post(
        f"/v1/files?scope_id={_SESSION_ID}",
        files={"file": (_FILE_NAME, io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
    )
    file_id = upload_resp.json()["id"]

    del_resp = files_client.delete(_file_url(file_id, _SESSION_ID))
    assert del_resp.status_code == 200
    deleted = DeletedFile.model_validate(del_resp.json())
    assert deleted.id == file_id
    assert deleted.type == "file_deleted"

    assert files_client.get(_file_url(file_id, _SESSION_ID)).status_code == 404


def test_delete_output_returns_404(files_client: TestClient, volume_root: Path) -> None:
    output_path = volume_root / "sessions" / _SESSION_ID / "outputs" / "result.csv"
    output_path.write_bytes(b"a,b\n1,2")
    output_id = str(output_path.resolve())

    resp = files_client.delete(_file_url(output_id, _SESSION_ID))
    assert resp.status_code == 404
