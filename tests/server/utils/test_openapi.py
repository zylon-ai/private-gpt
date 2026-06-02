from typing import Any

import pytest
from fastapi.testclient import TestClient

from private_gpt.settings.settings import settings


def _get_openapi_spec(test_client: TestClient) -> dict[str, Any]:
    prefix_path = settings().server.root_path
    response = test_client.get(f"{prefix_path}/openapi.json")
    assert (
        response.status_code == 200
    ), f"Failed to fetch OpenAPI schema: {response.status_code}"
    spec: dict[str, Any] = response.json()
    return spec


def test_all_non_hidden_endpoints_have_basic_properties(
    test_client: TestClient,
) -> None:
    errors: list[str] = []

    def check_doc(check: bool, message: str) -> None:
        if not check:
            errors.append(message)

    spec = _get_openapi_spec(test_client)
    paths: dict[str, Any] | None = spec.get("paths")

    if paths is None:
        pytest.fail("OpenAPI spec is missing 'paths' section")

    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue

        for method, operation in path_item.items():
            if not isinstance(operation, dict):
                continue

            hidden_config: dict[str, Any] | None = operation.get("hidden")
            if hidden_config and hidden_config.get("value") is True:
                continue

            check_doc(
                "tags" in operation, f"Path {path}, method {method} is missing tags"
            )
            check_doc(
                "summary" in operation,
                f"Path {path}, method {method} is missing summary",
            )
            check_doc(
                "description" in operation,
                f"Path {path}, method {method} is missing description",
            )

            responses: dict[str, Any] | None = operation.get("responses")
            check_doc(
                responses is not None,
                f"Path {path}, method {method} is missing responses",
            )

            if responses:
                for status_code, response in responses.items():
                    if not isinstance(response, dict):
                        continue

                    check_doc(
                        "description" in response,
                        f"Path {path}, method {method} is missing response description for status code {status_code}",
                    )

                    response_desc: str | None = response.get("description")
                    check_doc(
                        response_desc is not None and response_desc.strip() != "",
                        f"Path {path}, method {method} has empty response description for status code {status_code}",
                    )

            request_body: dict[str, Any] | None = operation.get("requestBody")
            mutating_methods: list[str] = ["post", "put", "patch"]

            if method.lower() in mutating_methods:
                check_doc(
                    request_body is not None,
                    f"Path {path}, method {method} is missing requestBody",
                )

            if request_body:
                check_doc(
                    "description" in request_body,
                    f"Path {path}, method {method} is missing requestBody description",
                )

                request_body_desc: str | None = request_body.get("description")
                check_doc(
                    request_body_desc is not None and request_body_desc.strip() != "",
                    f"Path {path}, method {method} has empty requestBody description",
                )

    if errors:
        pytest.fail("OpenAPI spec validation failed:\n" + "\n".join(errors))


def test_all_schemas_have_required_properties(test_client: TestClient) -> None:
    errors: list[str] = []

    def check_schema(check: bool, message: str) -> None:
        if not check:
            errors.append(message)

    def is_multipart_form_schema(schema_name: str) -> bool:
        return schema_name.startswith("Body_") and (
            "_post" in schema_name.lower()
            or "_put" in schema_name.lower()
            or "_patch" in schema_name.lower()
        )

    response = test_client.get("/openapi.json")
    assert (
        response.status_code == 200
    ), f"Failed to fetch OpenAPI schema: {response.status_code}"

    spec: dict[str, Any] = response.json()
    components: dict[str, Any] | None = spec.get("components")

    if components is None:
        pytest.fail("OpenAPI spec is missing 'components' section")

    schemas: dict[str, Any] | None = components.get("schemas")

    if schemas is None:
        pytest.fail("OpenAPI spec is missing 'schemas' section")

    for schema_name, schema_def in schemas.items():
        if not isinstance(schema_def, dict):
            continue

        if "error" in schema_name.lower():
            continue

        if is_multipart_form_schema(schema_name):
            # Skip multipart form body schemas - they can't have descriptions
            continue

        check_schema(
            "description" in schema_def, f"Schema {schema_name} is missing description"
        )

        description: str | None = schema_def.get("description")
        check_schema(
            description is not None and description.strip() != "",
            f"Schema {schema_name} has empty description",
        )

        check_schema(
            "type" in schema_def
            or "$ref" in schema_def
            or "allOf" in schema_def
            or "oneOf" in schema_def
            or "anyOf" in schema_def,
            f"Schema {schema_name} is missing type definition",
        )

        properties: dict[str, Any] | None = schema_def.get("properties")
        if properties:
            for prop_name, prop_def in properties.items():
                if not isinstance(prop_def, dict):
                    continue

                check_schema(
                    "type" in prop_def
                    or "$ref" in prop_def
                    or "allOf" in prop_def
                    or "oneOf" in prop_def
                    or "anyOf" in prop_def,
                    f"Schema {schema_name}, property {prop_name} is missing type definition",
                )

                check_schema(
                    "description" in prop_def,
                    f"Schema {schema_name}, property {prop_name} is missing description",
                )

                prop_description: str | None = prop_def.get("description")
                check_schema(
                    prop_description is not None and prop_description.strip() != "",
                    f"Schema {schema_name}, property {prop_name} has empty description",
                )

        if schema_def.get("type") == "array":
            items: dict[str, Any] | None = schema_def.get("items")
            check_schema(
                items is not None,
                f"Schema {schema_name} is array type but missing 'items' definition",
            )

            if items:
                check_schema(
                    "type" in items or "$ref" in items,
                    f"Schema {schema_name} array items missing type definition",
                )

    if errors:
        pytest.fail("Schema validation failed:\n" + "\n".join(errors))
