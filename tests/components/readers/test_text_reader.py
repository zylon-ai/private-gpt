import asyncio
import uuid
from pathlib import Path

from llama_index.core.schema import MetadataMode

from private_gpt.components.ingest.metadata_helper import MetadataKeys
from private_gpt.components.ingest.utils import convert_unsupported_file, get_file_info
from private_gpt.components.readers import ReaderComponent
from private_gpt.di import get_global_injector

injector = get_global_injector()
reader_component = injector.get(ReaderComponent)
reader = reader_component.get_reader("text")

TEST_FOLDER_PATH = Path(__file__).parents[0]
TEST_FILE_PATH = TEST_FOLDER_PATH / "files"
mock_extra_info = {
    MetadataKeys.ARTIFACT_ID.value: str(uuid.uuid4()),
    MetadataKeys.COLLECTION.value: str(uuid.uuid4()),
}


async def collect_nodes(file_info, reader_override=None):
    active_reader = reader_override or reader
    return [
        node
        async for node in active_reader.lazy_load_data(
            file_info=file_info,
            extra_info=mock_extra_info,
        )
    ]


def test_html_case() -> None:
    file = TEST_FILE_PATH / "file.html"
    mock_file_info = get_file_info(file, "file_image.html")

    r = reader_component.get_reader_by_extension(".html")
    nodes = asyncio.run(collect_nodes(mock_file_info, r))
    assert len(nodes) > 1, "Expected more than 1 node for the image case."

    markdown_content = "".join([node.get_content(MetadataMode.NONE) for node in nodes])
    assert "Comprehensive Profile of" in markdown_content, (
        "Expected an header with bold in the content."
    )
    assert "# Comprehensive Profile of [Alex Morgan]" in markdown_content, (
        "Expected a markdown header."
    )


def test_non_utf_8_html_case() -> None:
    file = TEST_FILE_PATH / "file_no_utf8.html"
    mock_file_info = convert_unsupported_file(get_file_info(file, "file_no_utf8.html"))

    nodes = asyncio.run(collect_nodes(mock_file_info))
    assert len(nodes) > 1, "Expected more than 1 node for the image case."

    markdown_content = "".join([node.get_content(MetadataMode.NONE) for node in nodes])
    assert "The fines collected pursuant to this Res" in markdown_content, (
        "Expected an header with bold in the content."
    )

    file = TEST_FILE_PATH / "another_file_no_utf8.html"
    mock_file_info = convert_unsupported_file(
        get_file_info(file, "another_file_no_utf8.html")
    )

    nodes = asyncio.run(collect_nodes(mock_file_info))
    assert len(nodes) > 1, "Expected more than 1 node for the image case."

    markdown_content = "".join([node.get_content(MetadataMode.NONE) for node in nodes])
    assert "Conflict of Interest" in markdown_content, (
        "Expected an header with bold in the content."
    )

    file = TEST_FILE_PATH / "file_window.html"
    mock_file_info = convert_unsupported_file(get_file_info(file, "file_window.html"))

    nodes = asyncio.run(collect_nodes(mock_file_info))
    assert len(nodes) > 1, "Expected more than 1 node for the image case."

    markdown_content = "".join([node.get_content(MetadataMode.NONE) for node in nodes])
    assert "the Director General of the Dubai" in markdown_content, (
        "Expected an header with bold in the content."
    )

    file = TEST_FILE_PATH / "file_unknown.txt"
    mock_file_info = convert_unsupported_file(get_file_info(file, "file_unknown.text"))

    nodes = asyncio.run(collect_nodes(mock_file_info))
    assert len(nodes) > 1, "Expected more than 1 node for the image case."

    markdown_content = "".join([node.get_content(MetadataMode.NONE) for node in nodes])
    assert "The Project Gutenberg eBook of The Great Gatsby" in markdown_content, (
        "Expected an header with bold in the content."
    )


def test_emoji_html_case() -> None:
    file = TEST_FILE_PATH / "file_emoji.html"
    mock_file_info = convert_unsupported_file(get_file_info(file, "file_emoji.html"))

    nodes = asyncio.run(collect_nodes(mock_file_info))
    assert len(nodes) > 1, "Expected more than 1 node for the image case."

    markdown_content = "".join([node.get_content(MetadataMode.NONE) for node in nodes])
    assert "blue star" in markdown_content, (
        "Expected an header with bold in the content."
    )
    assert "/_/" in markdown_content, (
        "We expect to process emoji images in the right way"
    )


def test_txt_case() -> None:
    file = TEST_FILE_PATH / "file.txt"
    mock_file_info = get_file_info(file, "file.txt")

    nodes = asyncio.run(collect_nodes(mock_file_info))
    assert len(nodes) > 1, "Expected more than 1 node."

    markdown_content = "".join([node.get_content(MetadataMode.NONE) for node in nodes])
    assert "Other files in this director" in markdown_content, (
        "Expected the test file content."
    )


def test_csv_case() -> None:
    file = TEST_FILE_PATH / "file.csv"
    mock_file_info = get_file_info(file, "file.csv")

    csv_reader = reader_component.get_reader("text", extension=".csv")
    nodes = asyncio.run(collect_nodes(mock_file_info, csv_reader))
    assert len(nodes) > 1, "Expected more than 1 node."

    markdown_content = "".join([node.get_content(MetadataMode.NONE) for node in nodes])
    assert "| Column1 | Column2 | Column3 | Column4 | Column5 |" in markdown_content, (
        "Expected a table in the second node."
    )


def test_csv_comma() -> None:
    file = TEST_FILE_PATH / "file_comma.csv"
    mock_file_info = get_file_info(file, "file_comma.csv")

    csv_reader = reader_component.get_reader("text", extension=".csv")
    nodes = asyncio.run(collect_nodes(mock_file_info, csv_reader))
    assert len(nodes) > 1, "Expected more than 1 node."

    markdown_content = "".join([node.get_content(MetadataMode.NONE) for node in nodes])
    assert "| Name | Age | City |" in markdown_content, (
        "Expected a table in the first node."
    )
    assert "| Alice | 30 | New York |" in markdown_content, (
        "Expected a table in the second node."
    )


def test_csv_semicolon() -> None:
    file = TEST_FILE_PATH / "file_semicolon.csv"
    mock_file_info = get_file_info(file, "file_semicolon.csv")

    csv_reader = reader_component.get_reader("text", extension=".csv")
    nodes = asyncio.run(collect_nodes(mock_file_info, csv_reader))
    assert len(nodes) > 1, "Expected more than 1 node."

    markdown_content = "".join([node.get_content(MetadataMode.NONE) for node in nodes])
    assert "| Name | Age | City |" in markdown_content, (
        "Expected a table in the first node."
    )
    assert "| Alice | 30 | New York |" in markdown_content, (
        "Expected a table in the second node."
    )


def test_tsv_tab() -> None:
    file = TEST_FILE_PATH / "file_tab.tsv"
    mock_file_info = get_file_info(file, "file_tab.tsv")

    tsv_reader = reader_component.get_reader("text", extension=".tsv")
    nodes = asyncio.run(collect_nodes(mock_file_info, tsv_reader))
    assert len(nodes) > 1, "Expected more than 1 node."

    markdown_content = "".join([node.get_content(MetadataMode.NONE) for node in nodes])
    assert "| Name | Age | City |" in markdown_content, (
        "Expected a table in the first node."
    )
    assert "| Alice | 30 | New York |" in markdown_content, (
        "Expected a table in the second node."
    )


def test_psv_pipe() -> None:
    file = TEST_FILE_PATH / "file_pipe.psv"
    mock_file_info = get_file_info(file, "file_pipe.psv")

    psv_reader = reader_component.get_reader("text", extension=".psv")
    nodes = asyncio.run(collect_nodes(mock_file_info, psv_reader))
    assert len(nodes) > 1, "Expected more than 1 node."

    markdown_content = "".join([node.get_content(MetadataMode.NONE) for node in nodes])
    assert "| Name | Age | City |" in markdown_content, (
        "Expected a table in the first node."
    )
    assert "| Alice | 30 | New York |" in markdown_content, (
        "Expected a table in the second node."
    )


def test_quote_in_csv() -> None:
    file = TEST_FILE_PATH / "file_quote.csv"
    mock_file_info = get_file_info(file, "file_quote.psv")

    psv_reader = reader_component.get_reader("text", extension=".psv")
    nodes = asyncio.run(collect_nodes(mock_file_info, psv_reader))
    assert len(nodes) > 1, "Expected more than 1 node."

    markdown_content = "".join([node.get_content(MetadataMode.NONE) for node in nodes])
    assert "| item" in markdown_content, "Expected a table in the first node."


def test_eml() -> None:
    file = TEST_FILE_PATH / "file.eml"
    mock_file_info = get_file_info(file, "file.eml")

    eml_reader = reader_component.get_reader("text", extension=".eml")
    nodes = asyncio.run(collect_nodes(mock_file_info, eml_reader))
    assert len(nodes) > 1, "Expected more than 1 node."

    markdown_content = "".join([node.get_content(MetadataMode.NONE) for node in nodes])
    assert "Thanks for your time today." in markdown_content, (
        "Expected a table in the first node."
    )
