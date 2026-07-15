from pathlib import Path
from unittest.mock import patch

from private_gpt.components.ingest.ingest_helper import IngestionHelper
from private_gpt.components.ingest.progress.errors import IngestionValidationErrors
from private_gpt.components.ingest.utils import FileInfo, get_file_info

# Mock path to the test files
TEST_FOLDER_PATH = Path(__file__).parents[0]
TEST_FILE_PATH = TEST_FOLDER_PATH / "test.pdf"


def test_get_file_info_valid_pdf():
    """Test extracting file info from a valid PDF file."""
    with (
        patch("private_gpt.components.ingest.utils.get_filesize") as mock_filesize,
        patch(
            "private_gpt.components.ingest.utils.get_guest_mime_type"
        ) as mock_guest_mime,
        patch(
            "private_gpt.components.ingest.utils.get_actual_mime_type"
        ) as mock_actual_mime,
        patch("private_gpt.components.ingest.utils.detect_encoding") as mock_encoding,
        patch("private_gpt.components.ingest.utils.extract_config") as mock_config,
    ):
        mock_filesize.return_value = 1024  # 1KB
        mock_guest_mime.return_value = "application/pdf"
        mock_actual_mime.return_value = "application/pdf"
        mock_encoding.return_value = "utf-8"
        mock_config.return_value = {
            "pages": 10,
            "is_encrypted": False,
            "has_images": True,
            "has_forms": False,
            "has_annotations": False,
            "has_attachments": False,
            "special": False,
        }

        file_info = get_file_info(TEST_FILE_PATH, "testfile.pdf")

        assert isinstance(file_info, FileInfo)
        assert file_info.file_size == 1024
        assert file_info.guest_mime_type == "application/pdf"
        assert file_info.actual_mime_type == "application/pdf"
        assert file_info.config["pages"] == 10
        assert not file_info.config["is_encrypted"]
        assert file_info.config["has_images"] is True
        assert file_info.config["has_forms"] is False
        assert file_info.config["has_annotations"] is False
        assert file_info.config["has_attachments"] is False
        assert not file_info.config["special"]


def test_validate_file_info_real_pdf():
    """Test validating a real PDF file."""
    file_info = get_file_info(TEST_FILE_PATH, "test.pdf")

    errors, warnings = IngestionHelper.validate_file_info(file_info)

    assert len(errors) == 0  # Assuming the real test PDF is valid
    assert len(warnings) == 0


def test_validate_file_info_real_pdf_with_various_pages():
    """Test validating a real PDF file removing pages."""
    file_info = get_file_info(TEST_FILE_PATH, "test.pdf")

    # 1. Pages is well-calculated
    errors, warnings = IngestionHelper.validate_file_info(file_info)
    assert len(errors) == 0
    assert len(warnings) == 0

    # 2. Pages is null
    file_info.config["pages"] = None
    errors, warnings = IngestionHelper.validate_file_info(file_info)
    assert len(errors) == 0
    assert len(warnings) == 0

    # 3. Pages is 0
    file_info.config["pages"] = 0
    errors, warnings = IngestionHelper.validate_file_info(file_info)
    assert len(errors) == 0
    assert len(warnings) == 0

    # 4. Pages is negative
    file_info.config["pages"] = -5
    errors, warnings = IngestionHelper.validate_file_info(file_info)
    assert len(errors) == 0
    assert len(warnings) == 0

    # 4. Pages is non-integer
    file_info.config["pages"] = "non-integer"
    errors, warnings = IngestionHelper.validate_file_info(file_info)
    assert len(errors) == 0
    assert len(warnings) == 0


def test_validate_file_info_real_encrypted_pdf():
    """Test file validation when using an encrypted PDF file."""
    # Assuming you provide an encrypted PDF for testing
    encrypted_pdf_path = TEST_FOLDER_PATH / "pdf_encrypted.pdf"
    file_info = get_file_info(encrypted_pdf_path, "pdf_encrypted.pdf")

    errors, warnings = IngestionHelper.validate_file_info(file_info)

    assert IngestionValidationErrors.SPECIAL_ENCRYPTED_FILE in warnings
    assert len(errors) == 0
    assert len(warnings) == 1
    assert file_info.config["is_encrypted"] is True


# TODO: disabled due to missing test file
# def test_validate_file_info_real_pdf_with_forms():
#     """Test file validation when the real PDF contains forms."""
#     # Assuming you provide a PDF with forms for testing
#     forms_pdf_path = TEST_FOLDER_PATH / "pdf_with_forms.pdf"
#     file_info = get_file_info(forms_pdf_path, "pdf_with_forms.pdf")
#
#     errors, warnings = IngestionHelper.validate_file_info(file_info)
#
#     assert IngestionValidationErrors.SPECIAL_FILE in warnings
#     assert len(errors) == 0
#     assert len(warnings) == 1
#     assert file_info.config["has_forms"] is True
#
#
# def test_validate_file_info_real_pdf_with_annotations():
#     """Test file validation when the real PDF contains annotations."""
#     # Assuming you provide a PDF with annotations for testing
#     annotations_pdf_path = TEST_FOLDER_PATH / "pdf_with_annotations.pdf"
#     file_info = get_file_info(annotations_pdf_path, "pdf_with_annotations.pdf")
#
#     errors, warnings = IngestionHelper.validate_file_info(file_info)
#
#     assert IngestionValidationErrors.SPECIAL_FILE in warnings
#     assert len(errors) == 0
#     assert len(warnings) == 1
#     assert file_info.config["has_annotations"] is True


def test_validate_file_info_encrypted_pdf():
    """Test file validation when the PDF is encrypted."""
    file_info = FileInfo(
        file_name="encryptedfile.pdf",
        extension=".pdf",
        file_data=TEST_FILE_PATH,
        guest_mime_type="application/pdf",
        actual_mime_type="application/pdf",
        file_size=1024,
        config={"pages": 5, "is_encrypted": True, "special": False},
    )

    errors, warnings = IngestionHelper.validate_file_info(file_info)

    assert IngestionValidationErrors.SPECIAL_ENCRYPTED_FILE in warnings
    assert len(errors) == 0
    assert len(warnings) == 1


def test_validate_file_info_invalid_pdf():
    """Test file validation when the PDF is invalid or malformed."""
    file_info = FileInfo(
        file_name="invalidfile.pdf",
        extension=".pdf",
        file_data=TEST_FILE_PATH,
        guest_mime_type="application/pdf",
        actual_mime_type="application/pdf",
        file_size=1024,
        config={"error": "Failed to parse PDF"},
    )

    errors, warnings = IngestionHelper.validate_file_info(file_info)

    assert IngestionValidationErrors.MALFORMED_FILE in errors
    assert len(errors) == 1
    assert len(warnings) == 0


# def test_validate_file_info_pdf_with_forms():
#     """Test file validation when the PDF contains forms."""
#     file_info = FileInfo(
#         file_name="pdfforms.pdf",
#         extension=".pdf",
#         file_data=TEST_FILE_PATH,
#         guest_mime_type="application/pdf",
#         actual_mime_type="application/pdf",
#         file_size=1024,
#         config={"pages": 5, "special": True, "is_encrypted": False},
#     )
#
#     errors, warnings = IngestionHelper.validate_file_info(file_info)
#
#     assert IngestionValidationErrors.SPECIAL_FILE in warnings
#     assert len(errors) == 0
#     assert len(warnings) == 1


def test_validate_file_info_file_without_extension():
    """Test file validation when the file has no extension."""
    file_info = FileInfo(
        file_name="filewithoutextension",
        extension=None,
        file_data=TEST_FILE_PATH,
        guest_mime_type=None,
        actual_mime_type="application/octet-stream",
        file_size=1024,
        config={"pages": 5},
    )

    errors, warnings = IngestionHelper.validate_file_info(file_info)

    assert IngestionValidationErrors.UNKNOWN_FILE_EXTENSION in errors
    assert len(errors) == 1
    assert len(warnings) == 0


def test_validate_file_info_empty_file():
    """Test file validation when the file is empty."""
    file_info = FileInfo(
        file_name="emptyfile.pdf",
        extension=".pdf",
        file_data=TEST_FILE_PATH,
        guest_mime_type="application/pdf",
        actual_mime_type="application/pdf",
        file_size=0,  # File size is zero, indicating an empty file
        config={},
    )

    errors, warnings = IngestionHelper.validate_file_info(file_info)

    assert IngestionValidationErrors.INVALID_FILE_SIZE in errors
    assert len(errors) == 1
    assert len(warnings) == 0
