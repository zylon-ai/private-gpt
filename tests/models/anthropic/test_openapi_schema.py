import copy
import json
from pathlib import Path
from typing import Any

import anthropic.types as sdk_types
import pytest

from tests.models.anthropic.registry import ALL_MAPPINGS, TypeMapping

OPENAPI_SPEC_URL = "https://storage.googleapis.com/stainless-sdk-openapi-specs/anthropic/anthropic-506a5ad71d522b4ae56ac3429380486647af1f92eddde80603480fb592d62b54.yml"
OPENAPI_DRIFT_WHITELIST_PATH = Path(__file__).with_name("openapi_drift_whitelist.json")


def _strip_zylon_fields(data: Any, zylon_fields: frozenset[str]) -> Any:
    if isinstance(data, dict):
        return {
            k: _strip_zylon_fields(v, zylon_fields)
            for k, v in data.items()
            if k not in zylon_fields
        }
    if isinstance(data, list):
        return [_strip_zylon_fields(item, zylon_fields) for item in data]
    return data


def _sdk_sample_stripped(mapping: TypeMapping) -> dict[str, Any]:
    payload = copy.deepcopy(mapping.sdk_sample)
    return _strip_zylon_fields(payload, mapping.zylon_only_fields)


_OPENAPI_NOISE_KEYS = frozenset(
    {
        "title",
        "description",
        "examples",
        "example",
        "deprecated",
        "default",
    }
)


def _resolve_schema_refs(
    openapi: dict[str, Any], node: Any, stack: tuple[str, ...] = ()
) -> Any:
    if isinstance(node, dict):
        if "propertyName" in node and "mapping" in node:
            # Mapping target component names can differ (RequestX vs X-Input)
            # while the discriminated variants are equivalent.
            node = {k: v for k, v in node.items() if k != "mapping"}

        ref = node.get("$ref")
        if isinstance(ref, str):
            if not ref.startswith("#/"):
                return node
            if ref in stack:
                return {"$ref": ref}

            cur: Any = openapi
            for part in ref[2:].split("/"):
                cur = cur[part]
            return _resolve_schema_refs(openapi, cur, (*stack, ref))

        return {
            k: _resolve_schema_refs(openapi, v, stack)
            for k, v in node.items()
            if k not in _OPENAPI_NOISE_KEYS and not k.startswith("x-")
        }

    if isinstance(node, list):
        return [_resolve_schema_refs(openapi, item, stack) for item in node]

    return node


def _normalize_schema(node: Any, parent_key: str | None = None) -> Any:
    if isinstance(node, dict):
        any_of = node.get("anyOf")
        if isinstance(any_of, list) and len(any_of) == 2:
            non_null = [item for item in any_of if item != {"type": "null"}]
            has_null = len(non_null) == 1
            if has_null:
                return _normalize_schema(non_null[0], parent_key)

    if isinstance(node, dict):
        normalized = {k: _normalize_schema(v, k) for k, v in node.items()}
        return {k: normalized[k] for k in sorted(normalized)}

    if isinstance(node, list):
        normalized_items = [_normalize_schema(item, parent_key) for item in node]

        if parent_key in {"required", "enum"}:
            try:
                return sorted(normalized_items)
            except TypeError:
                return normalized_items

        if parent_key in {"oneOf", "anyOf", "allOf"}:
            return sorted(
                normalized_items, key=lambda item: json.dumps(item, sort_keys=True)
            )

        return normalized_items

    return node


def _schema_diff(local_node: Any, remote_node: Any, path: str) -> list[str]:
    if type(local_node) is not type(remote_node):
        if (
            isinstance(local_node, (int, float))
            and isinstance(remote_node, (int, float))
            and float(local_node) == float(remote_node)
        ):
            return []
        return [
            f"{path}: type {type(local_node).__name__}!={type(remote_node).__name__}"
        ]

    if isinstance(local_node, dict):
        local_keys = set(local_node)
        remote_keys = set(remote_node)

        differences = [
            f"{path}.{k}: only_local" for k in sorted(local_keys - remote_keys)
        ]
        differences.extend(
            f"{path}.{k}: only_remote" for k in sorted(remote_keys - local_keys)
        )
        for key in sorted(local_keys & remote_keys):
            differences.extend(
                _schema_diff(local_node[key], remote_node[key], f"{path}.{key}")
            )
        return differences

    if isinstance(local_node, list):
        if path.endswith(".oneOf"):

            def _item_key(item: Any) -> str:
                if isinstance(item, dict):
                    t = item.get("properties", {}).get("type", {}).get("const")
                    if isinstance(t, str):
                        return f"type:{t}"
                    ref = item.get("$ref")
                    if isinstance(ref, str):
                        return f"ref:{ref}"
                return json.dumps(item, sort_keys=True)

            local_map = {_item_key(item): item for item in local_node}
            remote_map = {_item_key(item): item for item in remote_node}

            differences = [
                f"{path}[{k}]: only_local"
                for k in sorted(local_map.keys() - remote_map.keys())
            ]
            differences.extend(
                f"{path}[{k}]: only_remote"
                for k in sorted(remote_map.keys() - local_map.keys())
            )
            for key in sorted(local_map.keys() & remote_map.keys()):
                differences.extend(
                    _schema_diff(local_map[key], remote_map[key], f"{path}[{key}]")
                )
            return differences

        differences: list[str] = []
        if len(local_node) != len(remote_node):
            differences.append(f"{path}: len {len(local_node)}!={len(remote_node)}")
        for idx, (local_item, remote_item) in enumerate(
            zip(local_node, remote_node, strict=False)
        ):
            differences.extend(_schema_diff(local_item, remote_item, f"{path}[{idx}]"))
        return differences

    if (
        isinstance(local_node, (int, float))
        and isinstance(remote_node, (int, float))
        and float(local_node) == float(remote_node)
    ):
        return []

    if local_node != remote_node:
        return [f"{path}: {local_node!r}!={remote_node!r}"]

    return []


def _is_only_local_diff(diff: str) -> bool:
    return diff.endswith(": only_local")


def _is_globally_allowed_remote_diff(diff: str) -> bool:
    # Zylon supports dynamic model identifiers in routes where Anthropic
    # defines a stricter model union.
    if ".properties.model.anyOf: only_remote" in diff:
        return True
    # Citation payload shape is intentionally different in Zylon extensions.
    return ".citations" in diff or "citation_" in diff


def _load_openapi_drift_whitelist() -> dict[str, Any]:
    # NOTE:
    # This whitelist is the explicit contract for "allowed remote-only drift"
    # between our FastAPI schema and Anthropic's OpenAPI schema.
    #
    # Each entry in allowed_remote_diffs must explain:
    # 1) what differs (the exact schema-diff key), and
    # 2) why we intentionally allow it.
    #
    # Current intentional categories include:
    # - Zylon-specific request normalization (system/thinking/tool_choice shapes).
    # - Removed server-side custom tool_result variants (we keep only base tool_result).
    # - TextBlock optional/default behavior for streaming start blocks:
    #   local schema keeps text default="" so empty block starts can be emitted
    #   and then completed through deltas.
    with OPENAPI_DRIFT_WHITELIST_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def openapi_spec() -> dict[str, Any]:
    try:
        import requests
        import yaml

        response = requests.get(OPENAPI_SPEC_URL, timeout=20)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        text = response.text
        if "json" in content_type or text.lstrip().startswith("{"):
            import json

            return json.loads(text)
        return yaml.safe_load(text)

    except Exception as exc:
        pytest.skip(f"Anthropic OpenAPI spec unreachable, skipping: {exc}")


@pytest.fixture
def fastapi_openapi_spec(test_client) -> dict[str, Any]:
    response = test_client.get("/openapi.json")
    assert response.status_code == 200
    return response.json()


@pytest.fixture(scope="module")
def validate_against_schema(openapi_spec: dict[str, Any]):
    import jsonschema

    resolver = jsonschema.RefResolver(base_uri=OPENAPI_SPEC_URL, referrer=openapi_spec)
    schemas: dict[str, Any] = openapi_spec.get("components", {}).get("schemas", {})
    seen: set[str] = set()

    def _validate(payload: dict[str, Any], schema_name: str) -> bool:
        if schema_name in seen:
            return True

        if "Beta" in schema_name:
            return False

        found_candidate = False

        # The OpenAPI has several prefix/suffix in the names.
        # Just check all of them in order of priority
        for candidate in (
            schema_name,
            f"Request{schema_name}",
            f"Response{schema_name}",
        ):
            if candidate not in schemas:
                continue

            jsonschema.validate(
                instance=payload, schema=schemas[candidate], resolver=resolver
            )
            found_candidate = True
            break

        if not found_candidate:
            return False

        seen.add(schema_name)
        return True

    return _validate


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOpenAPISchemaCompatibility:
    def test_fastapi_openapi_contract_drift_fingerprint(
        self,
        openapi_spec: dict[str, Any],
        fastapi_openapi_spec: dict[str, Any],
    ) -> None:
        """Local schema may extend remote; remote drift must match the whitelist."""
        whitelist = _load_openapi_drift_whitelist()
        schema_pairs: dict[str, str] = whitelist["schema_pairs"]
        allowed_remote_diffs_by_schema: dict[str, dict[str, str]] = whitelist[
            "allowed_remote_diffs"
        ]

        remote_components = openapi_spec["components"]["schemas"]
        local_components = fastapi_openapi_spec["components"]["schemas"]
        failures: list[str] = []

        for local_name, remote_name in schema_pairs.items():
            local_schema = _normalize_schema(
                _resolve_schema_refs(fastapi_openapi_spec, local_components[local_name])
            )
            remote_schema = _normalize_schema(
                _resolve_schema_refs(openapi_spec, remote_components[remote_name])
            )
            diffs = set(_schema_diff(local_schema, remote_schema, local_name))
            actual_remote_diffs = sorted(
                d
                for d in diffs
                if not _is_only_local_diff(d)
                and not _is_globally_allowed_remote_diff(d)
            )
            allowed_remote_diffs = sorted(
                allowed_remote_diffs_by_schema.get(local_name, {}).keys()
            )
            unexpected_remote = sorted(
                set(actual_remote_diffs) - set(allowed_remote_diffs)
            )
            missing_remote = sorted(
                set(allowed_remote_diffs) - set(actual_remote_diffs)
            )

            if unexpected_remote or missing_remote:
                details: list[str] = []
                if unexpected_remote:
                    details.append(
                        "unexpected_remote:\n"
                        + (chr(10).join(unexpected_remote[:10]) or "<none>")
                    )
                if missing_remote:
                    details.append(
                        "missing_remote:\n"
                        + (chr(10).join(missing_remote[:10]) or "<none>")
                    )
                failures.append(
                    f"[{local_name}->{remote_name}] "
                    f"unexpected={len(unexpected_remote)} "
                    f"missing={len(missing_remote)}\n"
                    f"details:\n{chr(10).join(details) or '<none>'}"
                )

        assert not failures, (
            "OpenAPI contract drift changed unexpectedly.\n\n" + "\n\n".join(failures)
        )

    def test_endpoint_request_response_samples_pass_openapi_schema_validation(
        self,
        validate_against_schema: Any,
    ) -> None:
        """Validate endpoint-level request/response payloads used by Anthropic SDK."""
        samples: list[tuple[str, dict[str, Any], type | None]] = [
            (
                "CreateMessageParams",
                {
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 256,
                    "messages": [
                        {"role": "user", "content": "Summarize this text."},
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "Sure, share it."},
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Long text goes here"},
                            ],
                        },
                    ],
                    "system": "Be concise.",
                    "temperature": 0.4,
                    "stop_sequences": ["STOP"],
                    "tools": [
                        {
                            "name": "get_weather",
                            "description": "Get current weather by city",
                            "input_schema": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        }
                    ],
                    "tool_choice": {"type": "auto"},
                },
                None,
            ),
            (
                "CompletionRequest",
                {
                    "model": "claude-sonnet-4-6",
                    "prompt": "\n\nHuman: Say hello\n\nAssistant:",
                    "max_tokens_to_sample": 128,
                    "temperature": 0.2,
                    "top_k": 20,
                    "top_p": 0.9,
                    "stop_sequences": ["\n\nHuman:"],
                },
                None,
            ),
            (
                "CompletionResponse",
                {
                    "id": "compl_123",
                    "type": "completion",
                    "completion": "Hello!",
                    "stop_reason": "stop_sequence",
                    "model": "claude-sonnet-4-6",
                },
                sdk_types.Completion,
            ),
            (
                "ListResponse_ModelInfo_",
                {
                    "data": [
                        {
                            "id": "claude-sonnet-4-6",
                            "created_at": "2026-02-04T00:00:00Z",
                            "display_name": "Claude Sonnet 4.6",
                            "type": "model",
                            "max_tokens": 8192,
                            "max_input_tokens": 200000,
                            "capabilities": None,
                        }
                    ],
                    "first_id": "claude-sonnet-4-6",
                    "last_id": "claude-sonnet-4-6",
                    "has_more": False,
                },
                None,
            ),
            (
                "ModelInfo",
                {
                    "id": "claude-sonnet-4-6",
                    "created_at": "2026-02-04T00:00:00Z",
                    "display_name": "Claude Sonnet 4.6",
                    "type": "model",
                    "max_tokens": 8192,
                    "max_input_tokens": 200000,
                    "capabilities": None,
                },
                sdk_types.ModelInfo,
            ),
            (
                "CountMessageTokensParams",
                {
                    "model": "claude-sonnet-4-6",
                    "messages": [
                        {"role": "user", "content": "Count tokens for this input."},
                    ],
                    "system": "You are a tokenizer.",
                    "tools": [
                        {
                            "name": "get_weather",
                            "description": "Get current weather by city",
                            "input_schema": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        }
                    ],
                    "tool_choice": {"type": "auto"},
                },
                None,
            ),
            (
                "CountMessageTokensResponse",
                {"input_tokens": 42},
                sdk_types.MessageTokensCount,
            ),
        ]

        failures: list[str] = []
        for schema_name, payload, sdk_model in samples:
            try:
                validate_against_schema(payload, schema_name)
                if sdk_model is not None:
                    sdk_model.model_validate(payload)
            except Exception as exc:
                if "skip" in type(exc).__name__.lower():
                    raise
                failures.append(f"[{schema_name}] {type(exc).__name__}: {exc}")

        assert not failures, (
            "Endpoint request/response samples failed OpenAPI or SDK validation:\n\n"
            + "\n\n".join(failures)
        )

    def test_sdk_samples_pass_openapi_schema_validation(
        self,
        validate_against_schema: Any,
    ) -> None:
        failures: list[str] = []

        for mapping in ALL_MAPPINGS:
            payload = _sdk_sample_stripped(mapping)
            try:
                validate_against_schema(
                    payload, mapping.openapi_schema_name or mapping.sdk_schema_name
                )
            except Exception as exc:
                if "skip" in type(exc).__name__.lower():
                    raise
                failures.append(
                    f"[{mapping.sdk_schema_name}] {type(exc).__name__}: {exc}"
                )

        assert not failures, "OpenAPI schema validation failed:\n\n" + "\n\n".join(
            failures
        )

    def test_our_serialised_models_pass_openapi_schema_validation(
        self,
        validate_against_schema: Any,
    ) -> None:
        failures: list[str] = []

        for mapping in ALL_MAPPINGS:
            if mapping.our_type is None:
                continue

            try:
                instance = mapping.our_type.model_validate(mapping.sdk_sample)
                raw = instance.model_dump(by_alias=True, exclude_none=False)
                payload = _strip_zylon_fields(raw, mapping.zylon_only_fields)
                validate_against_schema(
                    payload, mapping.openapi_schema_name or mapping.sdk_schema_name
                )
            except Exception as exc:
                if "skip" in type(exc).__name__.lower():
                    raise
                failures.append(
                    f"[{mapping.our_type.__name__} → {mapping.sdk_schema_name}] "
                    f"{type(exc).__name__}: {exc}"
                )

        assert not failures, (
            "Serialised Zylon models failed OpenAPI schema validation:\n\n"
            + "\n\n".join(failures)
        )
