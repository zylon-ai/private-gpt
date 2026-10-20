import pytest

from private_gpt.components.environment.naming import unique_name


def _taken(*names: str):
    async def exists(candidate: str) -> bool:
        return candidate in names

    return exists


@pytest.mark.asyncio
async def test_free_name_passes_through() -> None:
    assert await unique_name("report.pdf", _taken()) == "report.pdf"


@pytest.mark.asyncio
async def test_taken_name_gets_the_first_suffix() -> None:
    assert await unique_name("report.pdf", _taken("report.pdf")) == "report (2).pdf"


@pytest.mark.asyncio
async def test_suffix_increments_until_free() -> None:
    exists = _taken("report.pdf", "report (2).pdf")
    assert await unique_name("report.pdf", exists) == "report (3).pdf"


@pytest.mark.asyncio
async def test_name_without_extension() -> None:
    assert await unique_name("README", _taken("README")) == "README (2)"


@pytest.mark.asyncio
async def test_only_the_last_extension_is_kept_apart() -> None:
    """Matches Windows: report.tar.gz → report.tar (2).gz."""
    taken = "report.tar.gz"
    assert await unique_name(taken, _taken(taken)) == "report.tar (2).gz"


@pytest.mark.asyncio
async def test_directory_prefix_is_preserved() -> None:
    taken = "/home/agent/workspace/report.md"
    assert (
        await unique_name(taken, _taken(taken)) == "/home/agent/workspace/report (2).md"
    )


@pytest.mark.asyncio
async def test_exhausting_max_attempts_raises() -> None:
    async def always_taken(candidate: str) -> bool:
        return True

    with pytest.raises(ValueError, match="after 3 attempts"):
        await unique_name("report.pdf", always_taken, max_attempts=3)
