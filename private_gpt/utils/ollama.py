import logging
from typing import Any, Generator, Mapping, Iterator
from tqdm import tqdm

try:
    from ollama import Client  # type: ignore
except ImportError as e:
    raise ImportError(
        "Ollama dependencies not found, install with `poetry install --extras llms-ollama or embeddings-ollama`"
    ) from e

logger = logging.getLogger(__name__)


def check_connection(client: Client) -> bool:
    try:
        client.list()
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e!s}")
        return False


def process_streaming(generator: Iterator[Mapping[str, Any]]) -> None:
    progress_bars = {}

    def create_progress_bar(total: int) -> tqdm:
        return tqdm(total=total, desc=f"Pulling model", unit='B', unit_scale=True)

    for chunk in generator:
        digest = chunk.get("digest")
        completed_size = chunk.get("completed", 0)
        total_size = chunk.get("total")

        if digest and total_size is not None:
            if digest not in progress_bars:
                progress_bars[digest] = create_progress_bar(total=total_size)

            progress_bar = progress_bars[digest]
            progress_bar.update(completed_size - progress_bar.n)

            if completed_size == total_size:
                progress_bar.close()
                del progress_bars[digest]

    # Close any remaining progress bars at the end
    for progress_bar in progress_bars.values():
        progress_bar.close()


def pull_model(client: Client, model_name: str, raise_error: bool = True) -> None:
    try:
        installed_models = [model["name"] for model in client.list().get("models", {})]
        if model_name not in installed_models:
            logger.info(f"Pulling model {model_name}. Please wait...")
            process_streaming(client.pull(model_name, stream=True))
            logger.info(f"Model {model_name} pulled successfully")
    except Exception as e:
        logger.error(f"Failed to pull model {model_name}: {e!s}")
        if raise_error:
            raise e
