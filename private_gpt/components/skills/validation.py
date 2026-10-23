def normalize_skill_metadata(value: object) -> dict[str, str] | None:
    """Validate metadata entries while preserving supported scalar coercion."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("metadata must be a mapping")

    metadata: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("metadata key must be a string")
        if not key.strip():
            raise ValueError("metadata keys must be non-empty")
        if not isinstance(item, (str, int, float, bool)):
            raise ValueError("metadata values must be strings, numbers, or booleans")
        metadata[key] = str(item)
    return metadata
