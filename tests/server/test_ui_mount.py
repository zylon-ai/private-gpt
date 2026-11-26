import pytest
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


def test_saved_workbench_web_choice_requires_approval_for_automatic_parallel(
    test_client: TestClient,
) -> None:
    playwright_api = pytest.importorskip("playwright.sync_api")
    html = test_client.get("/ui/").text
    origin = "https://workbench.test"
    automatic_provider = False

    with playwright_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except playwright_api.Error as exc:
            if "Executable doesn't exist" not in str(exc):
                raise
            try:
                browser = playwright.chromium.launch(channel="chrome")
            except playwright_api.Error as exc:
                if "not found" not in str(exc):
                    raise
                pytest.skip("Install Playwright Chromium to run the Workbench test")

        with browser:
            page = browser.new_page()
            page.add_init_script(
                """
                if (!localStorage.getItem("privategpt-workbench-state-v1")) {
                  localStorage.setItem("privategpt-workbench-state-v1", JSON.stringify({
                    privateGptBaseUrl: location.origin,
                    onboarding: {completed: true},
                    activeChatId: "legacy-chat",
                    models: [{id: "mock", capabilities: {}}],
                    selectedModel: "mock",
                    chats: [{id: "legacy-chat", title: "Saved chat", messages: [],
                             settings: {enabledWeb: true, enabledDocuments: false}}]
                  }));
                }
                """
            )

            def respond(route):
                url = route.request.url
                if not url.startswith(f"{origin}/"):
                    route.abort()
                elif url.endswith("/ui/"):
                    route.fulfill(body=html, content_type="text/html")
                elif url.endswith("/v1/tools/web-capabilities"):
                    route.fulfill(
                        json={
                            "provider": "parallel",
                            "automatic_provider": automatic_provider,
                            "search_available": True,
                            "fetch_available": False,
                        }
                    )
                elif url.endswith("/v1/messages"):
                    route.fulfill(
                        content_type="text/event-stream",
                        body='data: {"type":"content_block_start","index":0,'
                        '"content_block":{"type":"text","text":"OK"}}\n\n',
                    )
                else:
                    route.fulfill(json={"data": []})

            page.route("**/*", respond)

            def send(query: str) -> list[str]:
                page.locator("#composerInput").fill(query)
                with page.expect_request("**/v1/messages") as request:
                    page.get_by_role("button", name="Send", exact=True).click()
                page.wait_for_function("runtime.sendingChatIds.size === 0")
                return [tool["type"] for tool in request.value.post_data_json["tools"]]

            def show_web():
                page.locator("#composerAddMenuButton").click()
                return page.get_by_role("checkbox", name="Web · Parallel MCP")

            page.goto(f"{origin}/ui/#chat/legacy-chat")
            page.wait_for_function(
                "runtime.webCapabilities?.automatic_provider === false"
            )
            explicit_request = send("Use the explicitly configured search")

            automatic_provider = True
            page.reload()
            page.wait_for_function(
                "runtime.webCapabilities?.automatic_provider === true"
            )
            assert send("Check the saved choice before approval") == []
            assert explicit_request == ["explicit_parallel_web_search_v1"]
            web = show_web()
            assert not web.is_checked()
            assert "go to Parallel MCP" in page.locator("#toolsMenu").inner_text()
            web.check()
            assert send("Use automatic search after approval") == [
                "automatic_parallel_web_search_v1"
            ]

            page.reload()
            page.wait_for_function(
                "runtime.webCapabilities?.automatic_provider === true"
            )
            assert send("Use the persisted automatic approval") == [
                "automatic_parallel_web_search_v1"
            ]
            show_web().uncheck()
            assert send("Keep web search off") == []
