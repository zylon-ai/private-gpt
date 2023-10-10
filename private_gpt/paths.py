from pathlib import Path

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.settings.settings import settings

models_path: Path = PROJECT_ROOT_PATH / "models"
local_data_path: Path = PROJECT_ROOT_PATH / "local_data"
