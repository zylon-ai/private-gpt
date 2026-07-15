"""Initialization of PrivateGPT common to the main process and the celery worker."""
import logging
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path

import nltk
from llama_index.core import MockEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.global_handlers import create_global_handler
from llama_index.core.settings import Settings as LlamaIndexSettings

from private_gpt.paths import models_path
from private_gpt.settings.settings import Settings
from private_gpt.utils.dependencies import format_missing_dependency_message

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


ObservabilityProvider = Callable[["Settings"], None]

_PROVIDERS: dict[str, ObservabilityProvider] = {}


def register_observability(mode: str, provider: ObservabilityProvider) -> None:
    _PROVIDERS[mode] = provider


def _initialize_arize_phoenix(settings: "Settings") -> None:
    try:
        from openinference.instrumentation.llama_index import (  # ty:ignore[unresolved-import]
            LlamaIndexInstrumentor,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # ty:ignore[unresolved-import]
            OTLPSpanExporter,
        )
        from opentelemetry.sdk import trace as trace_sdk  # ty:ignore[unresolved-import]
        from opentelemetry.sdk.trace.export import (  # ty:ignore[unresolved-import]
            SimpleSpanProcessor,
        )
    except ImportError as e:
        raise ImportError(
            format_missing_dependency_message(
                "Arize Phoenix",
                extras="observability-arize-phoenix",
            )
        ) from e

    endpoint = f"{settings.phoenix.url}/v1/traces"
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

    LlamaIndexInstrumentor().instrument(
        tracer_provider=tracer_provider,
        use_legacy_callback_handler=False,
    )
    logging.getLogger("openinference.instrumentation.llama_index._handler").setLevel(
        logging.CRITICAL
    )
    logging.getLogger("opentelemetry.attributes").setLevel(logging.CRITICAL)


def _initialize_opik(settings: "Settings") -> None:
    try:
        import opik  # ty:ignore[unresolved-import]
        from opik.integrations.llama_index import (  # ty:ignore[unresolved-import]
            LlamaIndexCallbackHandler,
        )
    except ImportError as e:
        raise ImportError(
            format_missing_dependency_message(
                "Opik",
                extras="observability-opik",
            )
        ) from e

    opik.configure(
        api_key=settings.opik.api_key,
        workspace=settings.opik.workspace,
        url=settings.opik.host,
        use_local=bool(settings.opik.host and "comet" not in settings.opik.host),
        force=True,
    )
    opik_callback_handler = LlamaIndexCallbackHandler(
        project_name=settings.opik.project_name
    )
    LlamaIndexSettings.callback_manager = CallbackManager([opik_callback_handler])


def _initialize_simple(settings: "Settings") -> None:
    del settings
    logger.debug("Simple console logs observability mode")
    global_handler = create_global_handler("simple")
    if global_handler:
        LlamaIndexSettings.callback_manager = CallbackManager([global_handler])


_PROVIDERS.update(
    {
        "arize_phoenix": _initialize_arize_phoenix,
        "opik": _initialize_opik,
        "simple": _initialize_simple,
    }
)


def initialize_observability(settings: Settings) -> None:
    provider = _PROVIDERS.get(settings.observability.mode)
    if provider is None:
        logger.debug("No observability enabled")
        return
    provider(settings)
