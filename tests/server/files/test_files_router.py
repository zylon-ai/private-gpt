"""Direct router tests for POST/GET/DELETE /v1/files."""

import io
from pathlib import Path
from urllib.parse import quote

from fastapi.testclient import TestClient

from private_gpt.server.files.file_models import (
    DeletedFile,
    DeletedPrefix,
    FileListResponse,
    FileMetadata,
)
from private_gpt.server.files.file_service import _decode_file_id, _encode_file_id
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
    # ID is a base64-encoded canonical path pointing into uploads/
    decoded = _decode_file_id(meta.id)
    assert decoded.startswith("/")
    assert "uploads" in decoded
    assert decoded.endswith(_FILE_NAME)


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
    encoded_id = _encode_file_id("/nonexistent/path/file.txt")
    resp = files_client.get(_file_url(encoded_id, _SESSION_ID))
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
    output_path = volume_root / "outputs" / _SESSION_ID / "result.csv"
    output_path.write_bytes(_FILE_CONTENT)

    canonical = "/mnt/user-data/outputs/result.csv"
    file_id = _encode_file_id(canonical)

    resp = files_client.get(_file_url(file_id, _SESSION_ID, suffix="/content"))
    assert resp.status_code == 200
    assert resp.content == _FILE_CONTENT


def test_list_includes_outputs(files_client: TestClient, volume_root: Path) -> None:
    output_path = volume_root / "outputs" / _SESSION_ID / "result.png"
    output_path.write_bytes(b"\x89PNG")

    canonical = "/mnt/user-data/outputs/result.png"
    output_id = _encode_file_id(canonical)

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


def test_delete_output_file(files_client: TestClient, volume_root: Path) -> None:
    output_path = volume_root / "outputs" / _SESSION_ID / "result.csv"
    output_path.write_bytes(b"a,b\n1,2")

    canonical = "/mnt/user-data/outputs/result.csv"
    output_id = _encode_file_id(canonical)

    resp = files_client.delete(_file_url(output_id, _SESSION_ID))
    assert resp.status_code == 200
    assert resp.json()["id"] == output_id
    assert not output_path.exists()


def test_delete_workspace_file(files_client: TestClient, volume_root: Path) -> None:
    """A sandbox-generated workspace file can be removed after promotion."""
    workspace_path = volume_root / "user" / _SESSION_ID / "potato.md"
    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_path.write_bytes(b"# Potato")

    canonical = "/home/agent/workspace/potato.md"
    workspace_id = _encode_file_id(canonical)

    resp = files_client.delete(_file_url(workspace_id, _SESSION_ID))
    assert resp.status_code == 200, resp.text
    assert resp.json()["id"] == workspace_id
    assert not workspace_path.exists()


def test_list_namespaces(
    files_namespaces_client: TestClient,
) -> None:
    resp = files_namespaces_client.get("/v1/files/namespaces")
    assert resp.status_code == 200
    payload = resp.json()
    names = [item["name"] for item in payload["data"]]
    assert names == ["artifacts", "session", "skills"]
    modes = {item["name"]: item["default_mode"] for item in payload["data"]}
    assert modes == {
        "artifacts": "rw",
        "session": "rw",
        "skills": "ro",
    }
    assert all(item["root"].endswith(f"{item['name']}_ns") for item in payload["data"])


def test_list_namespaces_default_session(
    files_client: TestClient,
) -> None:
    """The default settings always register the implicit 'session' namespace."""
    resp = files_client.get("/v1/files/namespaces")
    assert resp.status_code == 200
    names = [item["name"] for item in resp.json()["data"]]
    assert "session" in names


# ---------------------------------------------------------------------------
# Custom-path uploads (object-storage style) and S3/blob-style PUT
# ---------------------------------------------------------------------------


def test_upload_with_custom_path(files_client: TestClient, volume_root: Path) -> None:
    """POST /v1/files accepts a custom object-storage-style key."""
    resp = files_client.post(
        f"/v1/files?scope_id={_SESSION_ID}&path=data/2024/report.pdf",
        files={"file": ("report.pdf", io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
    )
    assert resp.status_code == 200, resp.text
    meta = FileMetadata.model_validate(resp.json())
    decoded = _decode_file_id(meta.id)
    assert decoded == "/mnt/user-data/uploads/data/2024/report.pdf"
    assert meta.filename == "report.pdf"
    assert meta.downloadable is False
    assert meta.scope.id == _SESSION_ID

    stored = volume_root / "uploads" / _SESSION_ID / "data" / "2024" / "report.pdf"
    assert stored.read_bytes() == _FILE_CONTENT


def test_upload_with_uploads_prefix_is_normalized(
    files_client: TestClient, volume_root: Path
) -> None:
    """An explicit `uploads/` prefix on path is accepted and not doubled."""
    resp = files_client.post(
        f"/v1/files?scope_id={_SESSION_ID}&path=uploads/data/2024/report.pdf",
        files={"file": ("report.pdf", io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
    )
    assert resp.status_code == 200, resp.text
    decoded = _decode_file_id(resp.json()["id"])
    assert decoded == "/mnt/user-data/uploads/data/2024/report.pdf"

    stored = volume_root / "uploads" / _SESSION_ID / "data" / "2024" / "report.pdf"
    assert stored.read_bytes() == _FILE_CONTENT
    assert not (volume_root / "uploads" / _SESSION_ID / "uploads").exists()


def test_upload_with_invalid_path_rejected(files_client: TestClient) -> None:
    for bad in ("/abs/path.txt", "a/../b.txt", "dir/"):
        resp = files_client.post(
            f"/v1/files?scope_id={_SESSION_ID}&path={bad}",
            files={"file": (_FILE_NAME, io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
        )
        assert resp.status_code == 400, (bad, resp.text)


def test_upload_custom_path_appears_in_listing(
    files_client: TestClient,
) -> None:
    files_client.post(
        f"/v1/files?scope_id={_SESSION_ID}&path=data/2024/report.pdf",
        files={"file": ("report.pdf", io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
    )

    resp = files_client.get(f"/v1/files?scope_id={_SESSION_ID}")
    assert resp.status_code == 200
    listing = FileListResponse.model_validate(resp.json())
    ids = [_decode_file_id(f.id) for f in listing.data]
    assert "/mnt/user-data/uploads/data/2024/report.pdf" in ids


def test_delete_custom_path_file(files_client: TestClient) -> None:
    upload_resp = files_client.post(
        f"/v1/files?scope_id={_SESSION_ID}&path=data/2024/report.pdf",
        files={"file": ("report.pdf", io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
    )
    file_id = upload_resp.json()["id"]

    del_resp = files_client.delete(_file_url(file_id, _SESSION_ID))
    assert del_resp.status_code == 200, del_resp.text
    assert files_client.get(_file_url(file_id, _SESSION_ID)).status_code == 404


def test_put_object_s3_style(files_client: TestClient, volume_root: Path) -> None:
    """PUT /v1/files/{key} stores raw bytes at the key (S3/blob style)."""
    resp = files_client.put(
        f"/v1/files/data/2024/report.pdf?scope_id={_SESSION_ID}",
        content=_FILE_CONTENT,
        headers={"Content-Type": "text/csv"},
    )
    assert resp.status_code == 200, resp.text
    meta = FileMetadata.model_validate(resp.json())
    assert _decode_file_id(meta.id) == "/mnt/user-data/uploads/data/2024/report.pdf"
    assert meta.filename == "report.pdf"

    stored = volume_root / "uploads" / _SESSION_ID / "data" / "2024" / "report.pdf"
    assert stored.read_bytes() == _FILE_CONTENT

    # Round-trips through the rest of the API: metadata, content, listing.
    file_id = meta.id
    assert files_client.get(_file_url(file_id, _SESSION_ID)).status_code == 200
    content_resp = files_client.get(_file_url(file_id, _SESSION_ID, suffix="/content"))
    assert content_resp.status_code == 200
    assert content_resp.content == _FILE_CONTENT

    listing = FileListResponse.model_validate(
        files_client.get(f"/v1/files?scope_id={_SESSION_ID}").json()
    )
    assert any(_decode_file_id(f.id).endswith("report.pdf") for f in listing.data)


def test_put_object_uploads_prefix_normalized(
    files_client: TestClient, volume_root: Path
) -> None:
    resp = files_client.put(
        f"/v1/files/uploads/data/2024/report.pdf?scope_id={_SESSION_ID}",
        content=_FILE_CONTENT,
    )
    assert resp.status_code == 200, resp.text
    assert _decode_file_id(resp.json()["id"]) == (
        "/mnt/user-data/uploads/data/2024/report.pdf"
    )
    stored = volume_root / "uploads" / _SESSION_ID / "data" / "2024" / "report.pdf"
    assert stored.read_bytes() == _FILE_CONTENT


def test_put_object_invalid_path_rejected(files_client: TestClient) -> None:
    # Note: `..` segments are normalised away by the HTTP client before they
    # reach our handler, so the traversal case is only catchable via the
    # _normalize_upload_path unit test (see below).  What *is* catchable
    # at the HTTP level:
    for bad in ("//abs/path.txt", "dir/"):
        resp = files_client.put(
            f"/v1/files/{bad}?scope_id={_SESSION_ID}",
            content=_FILE_CONTENT,
        )
        assert resp.status_code == 400, (bad, resp.text)


def test_put_object_to_namespace(files_namespaces_client: TestClient) -> None:
    """PUT also works for non-session namespaces via PathResolver."""
    resp = files_namespaces_client.put(
        f"/v1/files/data/report.pdf?scope_id={_SESSION_ID}&namespace=artifacts",
        content=_FILE_CONTENT,
    )
    assert resp.status_code == 200, resp.text
    meta = FileMetadata.model_validate(resp.json())
    assert meta.id == "data/report.pdf"
    assert meta.namespace == "artifacts"

    meta_resp = files_namespaces_client.get(
        _file_url("data/report.pdf", _SESSION_ID) + "&namespace=artifacts"
    )
    assert meta_resp.status_code == 200, meta_resp.text
    assert FileMetadata.model_validate(meta_resp.json()).id == "data/report.pdf"


# ---------------------------------------------------------------------------
# Prefix listing (GET /v1/files?prefix=...)
# ---------------------------------------------------------------------------


def test_prefix_listing_filters_files(
    files_client: TestClient, volume_root: Path
) -> None:
    """Only files under the given prefix are returned."""
    for key in ("data/2024/jan.csv", "data/2024/feb.csv", "other/report.txt"):
        files_client.post(
            f"/v1/files?scope_id={_SESSION_ID}&path={key}",
            files={"file": (key.split("/")[-1], io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
        )

    resp = files_client.get(f"/v1/files?scope_id={_SESSION_ID}&prefix=data/2024/")
    assert resp.status_code == 200
    listing = FileListResponse.model_validate(resp.json())
    keys = [_decode_file_id(f.id) for f in listing.data]
    assert all("/data/2024/" in k for k in keys), keys
    assert not any("other" in k for k in keys), keys
    assert len(listing.data) == 2


def test_prefix_listing_no_prefix_returns_all(files_client: TestClient) -> None:
    for key in ("data/2024/jan.csv", "other/report.txt"):
        files_client.post(
            f"/v1/files?scope_id={_SESSION_ID}&path={key}",
            files={"file": (key.split("/")[-1], io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
        )

    resp = files_client.get(f"/v1/files?scope_id={_SESSION_ID}")
    listing = FileListResponse.model_validate(resp.json())
    assert len(listing.data) == 2


def test_prefix_listing_namespace(files_namespaces_client: TestClient) -> None:
    """Prefix filter also works for non-session namespaces."""
    files_namespaces_client.put(
        f"/v1/files/data/2024/jan.csv?scope_id={_SESSION_ID}&namespace=artifacts",
        content=_FILE_CONTENT,
    )
    files_namespaces_client.put(
        f"/v1/files/other/report.txt?scope_id={_SESSION_ID}&namespace=artifacts",
        content=_FILE_CONTENT,
    )

    resp = files_namespaces_client.get(
        f"/v1/files?scope_id={_SESSION_ID}&namespace=artifacts&prefix=data/"
    )
    assert resp.status_code == 200
    listing = FileListResponse.model_validate(resp.json())
    assert len(listing.data) == 1
    assert listing.data[0].id == "data/2024/jan.csv"


# ---------------------------------------------------------------------------
# Bulk delete by prefix (DELETE /v1/files?prefix=...)
# ---------------------------------------------------------------------------


def test_delete_by_prefix(files_client: TestClient) -> None:
    for key in ("data/2024/jan.csv", "data/2024/feb.csv", "other/report.txt"):
        files_client.post(
            f"/v1/files?scope_id={_SESSION_ID}&path={key}",
            files={"file": (key.split("/")[-1], io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
        )

    resp = files_client.delete(f"/v1/files?scope_id={_SESSION_ID}&prefix=data/2024/")
    assert resp.status_code == 200
    deleted = DeletedPrefix.model_validate(resp.json())
    assert deleted.prefix == "data/2024/"
    assert deleted.deleted_count == 2
    assert deleted.type == "prefix_deleted"

    listing = FileListResponse.model_validate(
        files_client.get(f"/v1/files?scope_id={_SESSION_ID}").json()
    )
    remaining_keys = [_decode_file_id(f.id) for f in listing.data]
    assert len(remaining_keys) == 1
    assert "other" in remaining_keys[0]


def test_delete_by_prefix_empty_prefix_rejected(files_client: TestClient) -> None:
    resp = files_client.delete(f"/v1/files?scope_id={_SESSION_ID}&prefix=")
    assert resp.status_code in (400, 422), resp.text


def test_delete_by_prefix_namespace(files_namespaces_client: TestClient) -> None:
    for key in ("data/2024/jan.csv", "data/2024/feb.csv"):
        files_namespaces_client.put(
            f"/v1/files/{key}?scope_id={_SESSION_ID}&namespace=artifacts",
            content=_FILE_CONTENT,
        )

    resp = files_namespaces_client.delete(
        f"/v1/files?scope_id={_SESSION_ID}&namespace=artifacts&prefix=data/"
    )
    assert resp.status_code == 200
    deleted = DeletedPrefix.model_validate(resp.json())
    assert deleted.deleted_count == 2


# ---------------------------------------------------------------------------
# HEAD /{file_id} — existence check with metadata headers
# ---------------------------------------------------------------------------


def test_head_file_returns_metadata_headers(files_client: TestClient) -> None:
    upload_resp = files_client.post(
        f"/v1/files?scope_id={_SESSION_ID}",
        files={"file": (_FILE_NAME, io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
    )
    file_id = upload_resp.json()["id"]
    encoded = quote(file_id, safe="")

    resp = files_client.request("HEAD", f"/v1/files/{encoded}?scope_id={_SESSION_ID}")
    assert resp.status_code == 200
    assert resp.headers["X-File-Filename"] == _FILE_NAME
    assert resp.headers["X-File-Mime-Type"] in (
        "text/plain",
        "text/csv",
    )  # libmagic varies
    assert int(resp.headers["X-File-Size"]) == len(_FILE_CONTENT)
    assert "X-File-Etag" in resp.headers
    assert "ETag" in resp.headers
    assert resp.content == b""


def test_head_file_not_found(files_client: TestClient) -> None:
    from private_gpt.server.files.file_service import _encode_file_id

    encoded = quote(_encode_file_id("/nonexistent/path.txt"), safe="")
    resp = files_client.request("HEAD", f"/v1/files/{encoded}?scope_id={_SESSION_ID}")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Content-Disposition inline vs attachment
# ---------------------------------------------------------------------------


def test_content_download_default_attachment(files_client: TestClient) -> None:
    upload_resp = files_client.post(
        f"/v1/files?scope_id={_SESSION_ID}",
        files={"file": (_FILE_NAME, io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
    )
    file_id = upload_resp.json()["id"]
    resp = files_client.get(_file_url(file_id, _SESSION_ID, suffix="/content"))
    assert resp.status_code == 200
    assert resp.headers["Content-Disposition"].startswith("attachment")


def test_content_download_inline(files_client: TestClient) -> None:
    upload_resp = files_client.post(
        f"/v1/files?scope_id={_SESSION_ID}",
        files={"file": (_FILE_NAME, io.BytesIO(_FILE_CONTENT), _MIME_TYPE)},
    )
    file_id = upload_resp.json()["id"]
    encoded = quote(file_id, safe="")
    resp = files_client.get(
        f"/v1/files/{encoded}/content?scope_id={_SESSION_ID}&disposition=inline"
    )
    assert resp.status_code == 200
    assert resp.headers["Content-Disposition"].startswith("inline")
    assert resp.content == _FILE_CONTENT
