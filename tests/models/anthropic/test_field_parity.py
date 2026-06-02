import typing
from typing import Any, get_args, get_origin

from tests.models.anthropic.registry import ALL_MAPPINGS, TypeMapping

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_field_names(model: Any) -> set[str]:
    """Collect field names, serialization aliases, and validation aliases."""
    names: set[str] = set()
    for fname, finfo in model.model_fields.items():
        names.add(fname)
        alias = getattr(finfo, "serialization_alias", None)
        if isinstance(alias, str):
            names.add(alias)
        val_alias = getattr(finfo, "validation_alias", None)
        if isinstance(val_alias, str):
            names.add(val_alias)
    return names


def _required_sdk_fields(mapping: TypeMapping) -> set[str]:
    if not mapping.sdk_type:
        return set()

    return {
        name
        for name, finfo in mapping.sdk_type.model_fields.items()
        if finfo.is_required() and name not in mapping.sdk_only_fields
    }


def _optional_sdk_fields(mapping: TypeMapping) -> set[str]:
    if not mapping.sdk_type:
        return set()

    return {
        name
        for name, finfo in mapping.sdk_type.model_fields.items()
        if not finfo.is_required()
    }


def _our_field_names(mapping: TypeMapping) -> set[str]:
    if not mapping.our_type:
        return set()
    return set(mapping.our_type.model_fields.keys())  # type: ignore[union-attr]


def _sdk_field_names(mapping: TypeMapping) -> set[str]:
    if not mapping.sdk_type:
        return set()
    return set(mapping.sdk_type.model_fields.keys())


def _is_optional_annotation(annotation: Any) -> bool:
    """Return True if the annotation is ``T | None`` / ``Optional[T]``."""
    if get_origin(annotation) is typing.Union:
        return type(None) in get_args(annotation)
    return False


class TestFieldParity:
    def test_required_sdk_fields_present_in_our_models(self) -> None:
        failures: list[str] = []

        for mapping in ALL_MAPPINGS:
            if mapping.our_type is None or mapping.sdk_type is None:
                continue

            our_names = _all_field_names(mapping.our_type)
            required = _required_sdk_fields(mapping)
            missing = required - our_names - mapping.sdk_only_fields

            if missing:
                failures.append(
                    f"[{mapping.our_type.__name__} ↔ {mapping.sdk_type.__name__}] "
                    f"SDK required field(s) missing from our model: {sorted(missing)}"
                )

        assert not failures, (
            "Some of our models are missing fields that the Anthropic SDK "
            "marks as required.\n\n" + "\n".join(failures)
        )

    def test_no_conflicting_field_names(self) -> None:
        failures: list[str] = []

        for mapping in ALL_MAPPINGS:
            if mapping.our_type is None:
                continue

            sdk_fields = mapping.sdk_type.model_fields if mapping.sdk_type else {}
            our_fields = mapping.our_type.model_fields if mapping.our_type else {}

            for fname in set(sdk_fields) & set(our_fields):
                if fname in mapping.zylon_only_fields | mapping.sdk_only_fields:
                    continue

                sdk_optional = not sdk_fields[
                    fname
                ].is_required() or _is_optional_annotation(sdk_fields[fname].annotation)
                our_optional = not our_fields[
                    fname
                ].is_required() or _is_optional_annotation(our_fields[fname].annotation)

                # SDK required + ours optional → acceptable (we may be stricter).
                # SDK optional + ours required → problematic; API may omit it.
                if sdk_optional and not our_optional:
                    failures.append(
                        f"[{mapping.our_type.__name__}] field '{fname}': "
                        f"SDK marks it as optional but our model requires it."
                    )

        assert not failures, "\n".join(failures)

    def test_sdk_added_optional_fields_are_accounted_for(self) -> None:
        failures: list[str] = []

        for mapping in ALL_MAPPINGS:
            if mapping.our_type is None:
                continue

            our_names = _our_field_names(mapping)
            optional_sdk = _optional_sdk_fields(mapping)
            unaccounted = optional_sdk - our_names - mapping.sdk_only_fields

            if unaccounted:
                failures.append(
                    f"[{mapping.our_type.__name__} ↔ {mapping.sdk_type.__name__}] "
                    f"SDK added optional field(s) not in our model and not in "
                    f"sdk_only_fields: {sorted(unaccounted)}\n"
                    f"  → Either add the field to our model or list it in "
                    f"TypeMapping.sdk_only_fields."
                )

        assert (
            not failures
        ), "SDK optional fields are unaccounted for in the registry.\n\n" + "\n".join(
            failures
        )

    def test_our_extension_fields_are_declared_in_registry(self) -> None:
        failures: list[str] = []

        for mapping in ALL_MAPPINGS:
            if mapping.our_type is None or mapping.sdk_type is None:
                continue

            sdk_names = _sdk_field_names(mapping)
            our_names = _our_field_names(mapping)
            extras = our_names - sdk_names - mapping.zylon_only_fields

            if extras:
                failures.append(
                    f"[{mapping.our_type.__name__}] has field(s) not in SDK and "
                    f"not declared in zylon_only_fields: {sorted(extras)}\n"
                    f"  → Add them to TypeMapping.zylon_only_fields."
                )

        assert not failures, "Undeclared Zylon extension fields found.\n\n" + "\n".join(
            failures
        )
