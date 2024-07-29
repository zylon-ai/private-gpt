import logging

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


def pull_model(client: Client, model_name: str, raise_error: bool = True) -> None:
    try:
        installed_models = [model["name"] for model in client.list().get("models", {})]
        if model_name not in installed_models:
            logger.info(f"Pulling model {model_name}. Please wait...")
            client.pull(model_name)
            logger.info(f"Model {model_name} pulled successfully")
    except Exception as e:
        logger.error(f"Failed to pull model {model_name}: {e!s}")
        if raise_error:
            raise e
