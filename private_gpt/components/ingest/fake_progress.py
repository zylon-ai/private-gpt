import math


def calculate_validation_timing(
    file_size: int | None,
) -> tuple[float | None, float | None]:
    """Calculate interval and jitter for file validation based on file size.

    Args:
        file_size: File size in bytes

    Returns:
        Tuple of (interval, jitter) in seconds
    """
    if not file_size:
        return None, None

    # Convert to MB for easier calculations
    file_size_mb = max(file_size / (1024 * 1024), 1)

    # Logarithmic scaling for interval (0.7-3s)
    min_interval = 0.7
    max_interval = 3
    interval = min(max(0.7 + 0.8 * math.log(file_size_mb), min_interval), max_interval)

    # Logarithmic scaling for jitter (4-10s)
    min_jitter = 4
    max_jitter = 10.0
    jitter = min(max(4 + 0.5 * math.log(file_size_mb), min_jitter), max_jitter)

    return interval, jitter


def calculate_parsing_timing(
    file_size: int | None, pages: int = 1
) -> tuple[float | None, float | None]:
    """Calculate interval and jitter for parsing based on number of pages and file size.

    Args:
        pages: Number of pages in the document
        file_size: File size in bytes

    Returns:
        Tuple of (interval, jitter) in seconds
    """
    if not file_size or pages <= 0:
        return None, None

    # Base time per page (5-20s range)
    base_time_per_page = 5

    # File size factor (larger files might need more processing)
    file_size_mb = max(file_size / (1024 * 1024), 1)
    size_factor = math.log(file_size_mb, 10) / 2  # Logarithmic scaling

    # Calculate interval based on pages and size
    interval = min(base_time_per_page * (1 + size_factor), 20)

    # Jitter should be proportional to interval but not exceed it
    jitter = min(interval * 0.4, 8)

    return interval, jitter
