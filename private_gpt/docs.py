from typing import Any

from fastapi import FastAPI

TITLE = "Private-GPT API"
DESCRIPTION = """\
PrivateGPT -built by Zylon- is a production-ready AI project that allows you to ask questions about your documents using the power
of Large Language Models (LLMs), even in scenarios without an Internet connection. 100% private, no data leaves your
execution environment at any point."""


def configure_openapi(app: FastAPI) -> None:
    """Configure OpenAPI schema for the FastAPI app."""

    # Merge -Input and -Output from the OpenAPI schema
    def merge_input_output(openapi_schema: dict[str, Any]) -> dict[str, Any]:
        components = openapi_schema.get("components", {})
        schemas = components.get("schemas", {})

        schemas_to_replace = {}
        schemas_to_remove = []

        for schema_name in list(schemas.keys()):
            if schema_name.endswith("-Input"):
                base_name = schema_name.replace("-Input", "")
                schemas_to_replace[schema_name] = base_name
            if schema_name.endswith("-Output"):
                base_name = schema_name.replace("-Output", "")
                if schema_name in schemas:
                    schemas_to_remove.append(schema_name)
                schemas_to_replace[schema_name] = base_name

        for old_name, new_name in schemas_to_replace.items():
            if new_name not in schemas:
                schemas[new_name] = schemas.pop(old_name)

        for schema_name in schemas_to_remove:
            schemas.pop(schema_name, None)

        components["schemas"] = dict(sorted(schemas.items()))

        def update_refs(obj: Any) -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "$ref" and isinstance(value, str):
                        for old_name, new_name in schemas_to_replace.items():
                            if f"#/components/schemas/{old_name}" in value:
                                obj[key] = value.replace(
                                    f"#/components/schemas/{old_name}",
                                    f"#/components/schemas/{new_name}",
                                )
                    else:
                        update_refs(value)
            elif isinstance(obj, list):
                for item in obj:
                    update_refs(item)

        update_refs(openapi_schema)
        return openapi_schema

    def ensure_descriptions(openapi_schema: dict[str, Any]) -> dict[str, Any]:
        """Backfill missing OpenAPI descriptions with deterministic defaults."""
        paths = openapi_schema.get("paths", {})
        if isinstance(paths, dict):
            for path_item in paths.values():
                if not isinstance(path_item, dict):
                    continue
                for operation in path_item.values():
                    if not isinstance(operation, dict):
                        continue
                    summary = str(operation.get("summary") or "").strip()
                    if not str(operation.get("description") or "").strip():
                        operation["description"] = (
                            summary if summary else "No description provided."
                        )
                    request_body = operation.get("requestBody")
                    if (
                        isinstance(request_body, dict)
                        and not str(request_body.get("description") or "").strip()
                    ):
                        request_body["description"] = "Request payload."

        schemas = (
            openapi_schema.get("components", {}).get("schemas", {})
            if isinstance(openapi_schema.get("components"), dict)
            else {}
        )
        if isinstance(schemas, dict):
            for schema_name, schema_def in schemas.items():
                if not isinstance(schema_def, dict):
                    continue
                if not str(schema_def.get("description") or "").strip():
                    schema_def["description"] = f"{schema_name} schema."
                properties = schema_def.get("properties")
                if isinstance(properties, dict):
                    for prop_name, prop_def in properties.items():
                        if not isinstance(prop_def, dict):
                            continue
                        if not str(prop_def.get("description") or "").strip():
                            prop_def[
                                "description"
                            ] = f"{prop_name.replace('_', ' ')} field."
        return openapi_schema

    def uniquify_property_titles(openapi_schema: dict[str, Any]) -> dict[str, Any]:
        """Replace ambiguous property titles with schema-specific titles."""
        schemas = (
            openapi_schema.get("components", {}).get("schemas", {})
            if isinstance(openapi_schema.get("components"), dict)
            else {}
        )
        if not isinstance(schemas, dict):
            return openapi_schema

        for schema_name, schema_def in schemas.items():
            if not isinstance(schema_def, dict):
                continue
            properties = schema_def.get("properties")
            if not isinstance(properties, dict):
                continue
            for prop_name, prop_def in properties.items():
                if not isinstance(prop_def, dict):
                    continue
                if prop_def.get("title") == "Input":
                    prop_def["title"] = f"{schema_name}{prop_name.title()}"

        return openapi_schema

    openapi_schema = app.openapi()
    openapi_schema = merge_input_output(openapi_schema)
    openapi_schema = ensure_descriptions(openapi_schema)
    openapi_schema = uniquify_property_titles(openapi_schema)
    app.openapi_schema = openapi_schema
