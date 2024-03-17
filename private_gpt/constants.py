import os
from pathlib import Path

PROJECT_ROOT_PATH: Path = Path(__file__).parents[1]
script_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(script_dir, "static/checked")  # Actual upload path for uploaded file
UNCHECKED_DIR = os.path.join(script_dir, "static/unchecked")  # Actual upload path for uploaded file

OCR_UPLOAD = os.path.join(script_dir, 'uploads') # temporary upload path for scanned pdf file
