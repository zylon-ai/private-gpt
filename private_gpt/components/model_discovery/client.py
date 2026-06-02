from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from pydantic import ValidationError

from private_gpt.components.model_discovery.models import UnclassifiedModel

if TYPE_CHECKING:
    from private_gpt.chat.input_models import ModelCapabilitiesOutput, ModelInfoOutput

logger = logging.getLogger(__name__)


def _build_headers(api_key: str | None) -> dict[str, str]:
    api_key = api_key.strip() if api_key else None
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = (
            api_key if api_key.lower().startswith("bearer ") else f"Bearer {api_key}"
        )
    return headers


def _with_query_param(url: str, name: str, value: str) -> str:
    parts = urlsplit(url)
    query_params = [
        (key, val)
        for key, val in parse_qsl(parts.query, keep_blank_values=True)
        if key != name
    ]
    query_params.append((name, value))
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(query_params),
            parts.fragment,
        )
    )


def _extract_model_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return []

    data = payload.get("data")
    if isinstance(data, list):
        return data

    models = payload.get("models")
    if isinstance(models, list):
        return models

    return []


def extract_model_items(payload: Any) -> list[Any]:
    return _extract_model_items(payload)


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def positive_int(value: Any) -> int | None:
    int_value = _to_int(value)
    if int_value is None or int_value <= 0:
        return None
    return int_value


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)

    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, tz=UTC)

    if isinstance(value, str):
        timestamp = _to_int(value)
        if timestamp is not None:
            return datetime.fromtimestamp(timestamp, tz=UTC)

        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            pass

    return datetime(1970, 1, 1, tzinfo=UTC)


def _parse_capabilities(value: Any) -> ModelCapabilitiesOutput | None:
    if value is None:
        return None

    from private_gpt.chat.input_models import ModelCapabilitiesOutput

    try:
        return ModelCapabilitiesOutput.model_validate(value)
    except ValidationError:
        logger.debug("Ignoring unsupported model capabilities payload: %s", value)
        return None


def model_info_from_item(item: Any) -> ModelInfoOutput | None:
    if not isinstance(item, dict):
        return None

    from private_gpt.chat.input_models import ModelInfoOutput

    model_id = (
        item.get("id") or item.get("key") or item.get("name") or item.get("model")
    )
    if not isinstance(model_id, str) or not model_id.strip():
        return None

    return ModelInfoOutput(
        id=model_id,
        created_at=_parse_datetime(item.get("created_at") or item.get("created")),
        display_name=str(item.get("display_name") or item.get("name") or model_id),
        type="model",
        max_tokens=_to_int(item.get("max_tokens") or item.get("max_output_tokens")),
        max_input_tokens=_to_int(
            item.get("max_input_tokens")
            or item.get("context_window")
            or item.get("context_length")
            or item.get("max_context_tokens")
            or item.get("max_context_length")
            or item.get("max_model_len")
            or _get_nested(item, "meta", "n_ctx")
            or _get_nested(item, "meta", "n_ctx_train")
        ),
        embed_dim=positive_int(
            item.get("embed_dim")
            or item.get("embedding_dimension")
            or _get_nested(item, "meta", "n_embd")
        ),
        capabilities=_parse_capabilities(item.get("capabilities")),
    )


def _get_nested(item: dict[str, Any], *keys: str) -> Any:
    current: Any = item
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


@dataclass(frozen=True)
class DiscoveryHttpClient:
    api_base: str
    api_key: str | None
    timeout: float

    def get_json(self, endpoint: str) -> Any | None:
        url = self._url_for(endpoint)
        return self.get_json_url(url)

    def get_root_json(self, endpoint: str) -> Any | None:
        url = self._root_url_for(endpoint)
        return self.get_json_url(url)

    def get_json_url(self, url: str) -> Any | None:
        try:
            response = requests.get(
                url,
                headers=_build_headers(self.api_key),
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            logger.debug("Error fetching %s: %s", url, exc)
            return None

    def get_model_infos(
        self,
        *,
        endpoint: str = "/models",
        fetch_all_pages: bool = True,
    ) -> list[ModelInfoOutput]:
        return [
            model_info
            for item in self._fetch_model_items(
                endpoint=endpoint,
                fetch_all_pages=fetch_all_pages,
            )
            if (model_info := model_info_from_item(item)) is not None
        ]

    def _fetch_model_items(
        self,
        *,
        endpoint: str = "/models",
        fetch_all_pages: bool,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        first_url = self._url_for(endpoint)
        next_url: str | None = first_url

        while next_url is not None:
            try:
                response = requests.get(
                    next_url,
                    headers=_build_headers(self.api_key),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as exc:
                logger.warning("Error fetching models from %s: %s", next_url, exc)
                return items

            for item in _extract_model_items(payload):
                if isinstance(item, dict):
                    items.append(item)

            if (
                not fetch_all_pages
                or not isinstance(payload, dict)
                or not payload.get("has_more")
            ):
                break

            last_id = payload.get("last_id")
            next_url = (
                _with_query_param(first_url, "after_id", last_id) if last_id else None
            )

        return items

    def _url_for(self, endpoint: str) -> str:
        return f"{self.api_base.rstrip('/')}/{endpoint.lstrip('/')}"

    def _root_url_for(self, endpoint: str) -> str:
        parts = urlsplit(self.api_base)
        return urlunsplit(
            (
                parts.scheme,
                parts.netloc,
                f"/{endpoint.lstrip('/')}",
                "",
                "",
            )
        )

    def get_unclassified_models(
        self,
        *,
        fetch_all_pages: bool,
    ) -> tuple[UnclassifiedModel, ...]:
        raw_items = self._fetch_model_items(fetch_all_pages=fetch_all_pages)
        return tuple(
            UnclassifiedModel(model=info, raw=item)
            for item in raw_items
            if (info := model_info_from_item(item)) is not None
        )
