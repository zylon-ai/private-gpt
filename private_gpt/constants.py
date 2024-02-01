import os
from pathlib import Path

PROJECT_ROOT_PATH: Path = Path(__file__).parents[1]
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
