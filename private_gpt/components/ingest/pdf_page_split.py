from io import BytesIO
from pathlib import Path


def group_consecutive_pages(pages: list[int]) -> list[tuple[int, int]]:
    """Group a sorted list of 0-indexed pages into consecutive-page ranges.

    Each returned tuple is an inclusive ``(start, end)`` range. Isolated
    pages become a single-page range (``start == end``).
    """
    if not pages:
        return []

    groups: list[tuple[int, int]] = []
    start = pages[0]
    end = pages[0]

    for page in pages[1:]:
        if page == end + 1:
            end = page
            continue
        groups.append((start, end))
        start = page
        end = page

    groups.append((start, end))
    return groups


def extract_pdf_pages_bytes(file_path: Path, start: int, end: int) -> bytes:
    """Extract an inclusive 0-indexed page range into a standalone PDF.

    Builds a new PDF in memory containing only the requested pages, using
    ``pypdf``.
    """
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(file_path))
    writer = PdfWriter()
    for page_index in range(start, end + 1):
        writer.add_page(reader.pages[page_index])

    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
