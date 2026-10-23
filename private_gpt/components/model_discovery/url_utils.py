from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

OPENAI_API_HOST = "api.openai.com"


def _with_scheme(value: str) -> str:
    value = value.strip()
    if "://" in value:
        return value
    return f"https://{value}"


def normalize_api_base(value: str) -> str:
    parts = urlsplit(_with_scheme(value).rstrip("/"))
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            "",
            "",
        )
    )


def api_base_host(value: str | None) -> str | None:
    if not value:
        return None
    parts = urlsplit(_with_scheme(value))
    return parts.hostname.lower() if parts.hostname else None


def is_openai_api_base(value: str | None) -> bool:
    return api_base_host(value) == OPENAI_API_HOST
