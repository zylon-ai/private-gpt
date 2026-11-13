"""The shipped ``settings.yaml`` proxy block must feed the field the code reads.

``server.network.proxy.https_server`` is a computed property over the
``https_proxy`` field, and that property is the value the scraper hands to
Chromium (which ignores proxy environment variables), so a proxy block that
names the property instead of the underlying field drops ``HTTPS_PROXY``
silently: ``https_server`` and ``http_server`` both stay ``None`` and the
browser runs without a proxy.
"""

from pathlib import Path
from typing import Any

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.settings.settings import ProxySettings
from private_gpt.settings.yaml import load_yaml_with_envvars

_HTTPS_PROXY = "https://proxy.example.com:8443"


def _default_proxy_block() -> dict[str, Any]:
    """Load the shipped default ``settings.yaml`` proxy block with HTTPS_PROXY set."""
    settings_path = Path(PROJECT_ROOT_PATH) / "settings.yaml"
    # An explicit environ keeps the test independent of the ambient environment.
    environ = {"HTTP_PROXY": "", "HTTPS_PROXY": _HTTPS_PROXY}
    with settings_path.open("r", encoding="utf-8") as stream:
        raw = load_yaml_with_envvars(stream, environ=environ)
    return raw["server"]["network"]["proxy"]


def test_https_proxy_env_var_reaches_the_proxy_model() -> None:
    proxy = ProxySettings(**_default_proxy_block())

    assert proxy.https_proxy == _HTTPS_PROXY

    # This is the exact expression `build_scrape_config` evaluates before
    # passing a proxy server to the playwright scrape script.
    url_cfg = proxy.https_server or proxy.http_server
    assert url_cfg is not None, "no proxy server reaches the scraper"
    assert url_cfg.host == "proxy.example.com"
    assert url_cfg.port == 8443
