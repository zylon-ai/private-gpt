from pathlib import Path

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.settings.settings import settings

models_path: Path = PROJECT_ROOT_PATH / "models"
models_cache_path: Path = models_path / "cache"
local_data_path: Path = PROJECT_ROOT_PATH / settings.data.local_data_folder
