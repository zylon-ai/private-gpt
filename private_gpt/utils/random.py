import uuid


def generate_deterministic_uuid_from_seed(
    seed: str,
) -> uuid.UUID:
    # Create a namespace UUID (using namespace_dns as a random choice)
    namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

    # Create a UUID using the input string within the namespace
    # Set version to 4 for "random" UUID
    name_uuid = uuid.uuid5(namespace, seed)

    # Convert to bytes and modify to set the version bits to UUID v4
    uuid_bytes = bytearray(name_uuid.bytes)

    # Set the version bits to 4 for UUID v4
    uuid_bytes[6] = (uuid_bytes[6] & 0x0F) | 0x40

    # Set the variant bits to RFC 4122
    uuid_bytes[8] = (uuid_bytes[8] & 0x3F) | 0x80

    # Return the result as a UUID string
    return uuid.UUID(bytes=bytes(uuid_bytes))
