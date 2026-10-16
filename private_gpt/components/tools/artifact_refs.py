"""Artifact references passed by the model to knowledge tools.

Models often copy the artifact_id of a neighbouring entry from a file list
(UUIDs look alike). Each reference therefore carries the file name next to the
id, and the server rejects a pair that does not match, telling the model which
artifact_id belongs to the name it gave so it can fix the call.
"""

import asyncio
import posixpath
from collections.abc import Sequence
from typing import Annotated, Any

from llama_index.core.schema import BaseNode
from pydantic import BaseModel, Field, ValidationError, WithJsonSchema

from private_gpt.chat.extensions.context_filter import ContextFilter
from private_gpt.components.ingest.metadata_helper import MetadataKeys
from private_gpt.components.node_store.node_store_component import NodeStoreComponent

# Upper bound of nodes scanned when looking up the artifacts behind a file name.
_NAME_LOOKUP_LIMIT = 500
# Largest request scope whose file names are compared one by one as a fallback.
_SCOPE_SCAN_LIMIT = 200


class ArtifactRef(BaseModel):
    name: str = Field(description="The file name, exactly as listed")
    artifact_id: str = Field(description="The artifact_id listed next to that name")


def _artifact_ref_item_schema() -> dict[str, Any]:
    schema = ArtifactRef.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema


# Tool arguments reach the tool function without validation, so items may be
# plain strings or dicts. The JSON schema shown to the model is inlined (no
# $ref) because tool schemas are re-parsed by create_model_from_json_schema.
ArtifactRefInput = Annotated[
    ArtifactRef | dict[str, Any] | str,
    WithJsonSchema(_artifact_ref_item_schema()),
]

SHAPE_ERROR = (
    'Pass each file as {"name": ..., "artifact_id": ...}, '
    "copied from the same entry of the file list."
)


def _normalize(name: str) -> str:
    return name.strip().casefold()


def _basename(name: str) -> str:
    return posixpath.basename(_normalize(name).replace("\\", "/"))


def _names_match(given: str, actual: str) -> bool:
    return _normalize(given) == _normalize(actual) or _basename(given) == _basename(
        actual
    )


def _parse_refs(refs: Sequence[Any], action: str) -> list[ArtifactRef]:
    parsed: list[ArtifactRef] = []
    for ref in refs:
        if isinstance(ref, ArtifactRef):
            parsed.append(ref)
            continue
        if not isinstance(ref, dict):
            raise ValueError(f"artifacts rejected, nothing was {action}: {SHAPE_ERROR}")
        try:
            parsed.append(ArtifactRef.model_validate(ref))
        except ValidationError:
            raise ValueError(
                f"artifacts rejected, nothing was {action}: {SHAPE_ERROR}"
            ) from None
    return parsed


def _file_name(node: BaseNode) -> str | None:
    value = node.metadata.get(MetadataKeys.FILENAME.value)
    return str(value) if value else None


async def _names_by_id(
    collection: str,
    artifact_ids: list[str],
    node_store_component: NodeStoreComponent,
) -> dict[str, str | None]:
    """Map each known artifact_id to its file name; unknown ids are omitted."""

    def lookup(artifact_id: str) -> list[BaseNode]:
        return node_store_component.filtered_nodes(
            collection, artifacts=[artifact_id], limit=1
        )

    results = await asyncio.gather(
        *(asyncio.to_thread(lookup, artifact_id) for artifact_id in artifact_ids)
    )
    return {
        artifact_id: _file_name(nodes[0])
        for artifact_id, nodes in zip(artifact_ids, results, strict=True)
        if nodes
    }


async def _ids_by_name(
    collection: str,
    name: str,
    scope: list[str] | None,
    node_store_component: NodeStoreComponent,
) -> dict[str, str]:
    """Artifacts (artifact_id -> file name) whose file name matches `name`."""
    # Metadata filters are exact matches: try the name as given and its basename.
    stripped = name.strip()
    candidates = {stripped, posixpath.basename(stripped.replace("\\", "/"))}

    def lookup(file_name: str) -> list[BaseNode]:
        return node_store_component.filtered_nodes(
            collection,
            artifacts=scope,
            filter_dicts=[{"key": MetadataKeys.FILENAME.value, "value": file_name}],
            limit=_NAME_LOOKUP_LIMIT,
        )

    results = await asyncio.gather(
        *(asyncio.to_thread(lookup, c) for c in candidates if c)
    )
    matches: dict[str, str] = {}
    for nodes in results:
        for node in nodes:
            artifact_id = node.metadata.get(MetadataKeys.ARTIFACT_ID.value)
            file_name = _file_name(node)
            if artifact_id and file_name and _names_match(name, file_name):
                matches.setdefault(str(artifact_id), file_name)

    # Exact lookups miss case or folder differences; a bounded scope is small
    # enough to compare every file name directly.
    if not matches and scope and len(scope) <= _SCOPE_SCAN_LIMIT:
        names = await _names_by_id(collection, scope, node_store_component)
        matches = {
            artifact_id: file_name
            for artifact_id, file_name in names.items()
            if file_name and _names_match(name, file_name)
        }
    return matches


async def resolve_artifact_refs(
    refs: Sequence[Any] | None,
    context_filter: ContextFilter | None,
    node_store_component: NodeStoreComponent,
    action: str = "searched",
) -> list[str] | None:
    """Validate the model's artifact references and return their artifact_ids.

    Returns None when no restriction was requested. Raises ValueError with a
    model-facing message when any reference is malformed, unknown, outside the
    request scope, or pairs a name with another file's artifact_id.
    """
    if not refs:
        return None
    if isinstance(refs, str | dict):
        raise ValueError(f"artifacts rejected, nothing was {action}: {SHAPE_ERROR}")
    parsed = _parse_refs(refs, action)

    if not context_filter or not context_filter.collection:
        raise ValueError("collection is required in context")
    collection = context_filter.collection
    scope = list(dict.fromkeys(context_filter.artifacts or [])) or None

    ids = list(dict.fromkeys(ref.artifact_id.strip() for ref in parsed))
    known = await _names_by_id(collection, ids, node_store_component)

    problems: list[str] = []
    for ref in parsed:
        artifact_id = ref.artifact_id.strip()
        in_scope = scope is None or artifact_id in scope
        if artifact_id in known and in_scope:
            actual = known[artifact_id]
            # Files without a recorded name cannot be checked; trust the id.
            if actual is None or _names_match(ref.name, actual):
                continue

        if artifact_id not in known:
            passed = f"you passed {artifact_id}, which matches no file"
        elif not in_scope:
            passed = f"you passed {artifact_id}, which is not available here"
        else:
            passed = f"you passed {artifact_id}, which is '{known[artifact_id]}'"

        matches = await _ids_by_name(collection, ref.name, scope, node_store_component)
        if len(matches) == 1:
            (match_id, match_name), *_ = matches.items()
            problems.append(f"'{match_name}' has artifact_id {match_id}; {passed}")
        elif matches:
            listed = ", ".join(
                f"'{file_name}' (artifact_id {match_id})"
                for match_id, file_name in matches.items()
            )
            problems.append(
                f"several files are named '{ref.name}': {listed}; {passed}. "
                "Pick the one you mean"
            )
        else:
            problems.append(f"no file is named '{ref.name}'; {passed}")

    if problems:
        raise ValueError(
            f"artifacts rejected, nothing was {action}: "
            + "; ".join(problems)
            + ". Call again with the matching artifact_id."
        )
    return ids
