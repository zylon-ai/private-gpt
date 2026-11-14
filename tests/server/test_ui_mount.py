from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from private_gpt.launcher import UI_DIRECTORY, create_app
from tests.fixtures.mock_injector import MockInjector


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


@pytest.mark.parametrize("code_page", ["cp1252", "cp932"])
def test_ui_index_is_read_as_utf8_whatever_the_locale(
    applied_migrations: MockInjector,
    locale_encoding: Callable[[str], None],
    code_page: str,
) -> None:
    # The code pages of English and Japanese Windows.
    locale_encoding(code_page)
    client = TestClient(create_app(applied_migrations.test_injector))

    response = client.get("/ui/")

    index = (UI_DIRECTORY / "index.html").read_text(encoding="utf-8")
    non_ascii_line = next(line for line in index.splitlines() if not line.isascii())
    assert non_ascii_line in response.text
