"""Initialization of PrivateGPT common to the main process and the celery worker."""

import logging
import sys
import zipfile
from pathlib import Path

import nltk
from llama_index.core import MockEmbedding
from llama_index.core.settings import Settings as LlamaIndexSettings

from private_gpt.paths import models_path
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


def download_nltk_package_if_not_present(
    package_name: str, package_category: str, download_dir: str
) -> None:
    """If the required nlt package is not present, download it."""
    try:
        nltk.find(f"{package_category}/{package_name}", paths=[download_dir])
    except (LookupError, zipfile.BadZipFile, OSError):
        # NLTK may leave a broken zip behind after an interrupted download.
        # Remove the cached archive and extracted directory before retrying.
        zip_path = Path(download_dir) / package_category / f"{package_name}.zip"
        extract_dir = Path(download_dir) / package_category / package_name
        if zip_path.exists():
            zip_path.unlink()
        if extract_dir.exists():
            import shutil

            shutil.rmtree(extract_dir)
        nltk.download(package_name, download_dir=download_dir)
        # Some packages are downloaded as zip files and it doesn't get unzipped
        # https://github.com/nltk/nltk/issues/3028
        unzip_download_nltk_package_if_not_present(
            package_name=package_name,
            package_category=package_category,
            download_dir=download_dir,
        )


def unzip_download_nltk_package_if_not_present(
    package_name: str, package_category: str, download_dir: str
) -> None:
    zip_path = Path(download_dir) / package_category / f"{package_name}.zip"
    if zip_path.exists():
        extract_dir = zip_path.parent
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)


def initialize_globals() -> None:
    """Initialize global settings and dependencies."""
    # Set global embedding model to Mock to prevent LlamaIndex to default to use OpenAI
    LlamaIndexSettings.embed_model = MockEmbedding(384)

    # Install ingestion required dependencies
    # Prerequisite for Unstructured.io to work
    nltk_data_dir = str(models_path / "nltk_cache")
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)

    download_nltk_package_if_not_present(
        package_category="tokenizers",
        package_name="punkt_tab",
        download_dir=nltk_data_dir,
    )

    download_nltk_package_if_not_present(
        package_category="tokenizers", package_name="punkt", download_dir=nltk_data_dir
    )

    download_nltk_package_if_not_present(
        package_category="taggers",
        package_name="averaged_perceptron_tagger_eng",
        download_dir=nltk_data_dir,
    )

    download_nltk_package_if_not_present(
        package_category="taggers",
        package_name="averaged_perceptron_tagger",
        download_dir=nltk_data_dir,
    )

    download_nltk_package_if_not_present(
        package_category="corpora",
        package_name="stopwords",
        download_dir=nltk_data_dir,
    )

    download_nltk_package_if_not_present(
        package_category="corpora",
        package_name="wordnet",
        download_dir=nltk_data_dir,
    )

    # Increase the recursion limit to avoid stack overflow
    # in pypdf with table contents
    sys.setrecursionlimit(5000)


def initialize_observability(settings: Settings) -> None:
    """OpenTelemetry tracing to the configured exporters (``mode: otel``)."""
    if settings.observability.mode != "otel":
        logger.debug("No observability enabled")
        return
    from private_gpt.observability.setup import configure_tracing

    configure_tracing(settings)
