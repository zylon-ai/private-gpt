from typing import Any

from tests.models.anthropic.registry import ALL_MAPPINGS, TypeMapping


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


def _serialize(instance: Any) -> dict[str, Any]:
    if hasattr(instance, "model_dump"):
        return instance.model_dump(by_alias=True, exclude_none=True)
    return dict(instance)


def _round_trip_payload(mapping: TypeMapping) -> dict[str, Any]:
    our_instance = mapping.our_type.model_validate(mapping.sdk_sample)  # type: ignore[union-attr]
    serialised = _serialize(our_instance)
    return _strip_zylon_fields(serialised, mapping.zylon_only_fields)


class TestSDKParse:
    def test_our_models_are_parseable_by_sdk_types(self) -> None:
        failures: list[str] = []

        for mapping in ALL_MAPPINGS:
            if mapping.our_type is None:
                continue
            if mapping.sdk_type is None:
                continue

            try:
                payload = _round_trip_payload(mapping)
                parsed = mapping.sdk_type.model_validate(payload)

                our_type_val = mapping.sdk_sample.get("type")
                if our_type_val and hasattr(parsed, "type"):
                    assert parsed.type == our_type_val, (
                        f"[{mapping.our_type.__name__}] type discriminator "
                        f"mismatch after round-trip: "
                        f"expected={our_type_val!r} got={parsed.type!r}"
                    )
            except Exception as exc:
                failures.append(
                    f"[{mapping.our_type.__name__} → {mapping.sdk_type.__name__}] "
                    f"{type(exc).__name__}: {exc}"
                )

        assert (
            not failures
        ), "SDK parse failed for the following models:\n\n" + "\n\n".join(failures)

    def test_sdk_samples_are_parseable_by_our_models(self) -> None:
        failures: list[str] = []
        for mapping in ALL_MAPPINGS:
            if mapping.our_type is None:
                continue
            if mapping.sdk_type is None:
                continue

            try:
                mapping.our_type.model_validate(mapping.sdk_sample)
            except Exception as exc:
                failures.append(
                    f"[{mapping.sdk_type.__name__} → {mapping.our_type.__name__}] "
                    f"{type(exc).__name__}: {exc}"
                )

        assert (
            not failures
        ), "Our models rejected valid SDK payloads:\n\n" + "\n\n".join(failures)

    def test_sdk_sample_directly_parseable_by_sdk_type(self) -> None:
        failures: list[str] = []

        for mapping in ALL_MAPPINGS:
            try:
                if mapping.sdk_type is None:
                    continue
                mapping.sdk_type.model_validate(mapping.sdk_sample)
            except Exception as exc:
                failures.append(
                    f"[{mapping.sdk_type.__name__}] sdk_sample is invalid "
                    f"according to the SDK type itself: "
                    f"{type(exc).__name__}: {exc}"
                )

        assert (
            not failures
        ), "Registry sdk_sample payloads are invalid:\n\n" + "\n\n".join(failures)
