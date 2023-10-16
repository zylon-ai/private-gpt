from pathlib import Path

from private_gpt.constants import PROJECT_ROOT_PATH

models_path: Path = PROJECT_ROOT_PATH / "models"
models_cache_path: Path = models_path / "cache"
local_data_path: Path = PROJECT_ROOT_PATH / "local_data"
