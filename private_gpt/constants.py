import os
from pathlib import Path

PROJECT_ROOT_PATH: Path = Path(__file__).parents[1]
script_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(script_dir, "static")
