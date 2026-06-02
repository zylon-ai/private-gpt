from __future__ import annotations

from typing import TYPE_CHECKING

from private_gpt.components.model_discovery.client import DiscoveryHttpClient
from private_gpt.components.model_discovery.models import (
    ModelDiscoveryResult,
)
from private_gpt.components.model_discovery.strategies import StrategyChain
from private_gpt.components.model_discovery.url_utils import normalize_api_base

if TYPE_CHECKING:
    from private_gpt.components.model_discovery.models import ModelKind

DEFAULT_DISCOVERY_TIMEOUT = 3.0


def are_distinct_api_bases(first: str | None, second: str | None) -> bool:
    if not first or not second:
        return False
    return normalize_api_base(first) != normalize_api_base(second)


def discover_model_infos(
    api_base: str,
    api_key: str | None,
    *,
    force_kind: ModelKind | None = None,
    timeout: float = DEFAULT_DISCOVERY_TIMEOUT,
    fetch_all_pages: bool = True,
) -> ModelDiscoveryResult:
    client = _build_client(api_base, api_key, timeout)
    classification = StrategyChain().discover(
        client,
        fetch_all_pages=fetch_all_pages,
        force_kind=force_kind,
    )
    return ModelDiscoveryResult.from_classified(
        classification.provider,
        classification.models,
    )


def _build_client(
    api_base: str,
    api_key: str | None,
    timeout: float,
) -> DiscoveryHttpClient:
    return DiscoveryHttpClient(api_base=api_base, api_key=api_key, timeout=timeout)
