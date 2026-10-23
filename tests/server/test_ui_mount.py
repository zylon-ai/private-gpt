from fastapi.testclient import TestClient


def test_ui_path_still_serves_bundled_ui_index(test_client: TestClient) -> None:
    response = test_client.get("/ui/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'const DEFAULT_BASE_URL = "http://127.0.0.1:8080";' in response.text


def test_ui_path_without_trailing_slash_redirects_to_index(
    test_client: TestClient,
) -> None:
    response = test_client.get("/ui", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/ui/"
