import os
from pathlib import Path

PROJECT_ROOT_PATH: Path = Path(__file__).parents[1]
script_dir = os.path.dirname(os.path.abspath(__file__))
# Create directories if they don't exist
UPLOAD_DIR = os.path.join(script_dir, "static/checked")
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Actual upload path for uploaded file

UNCHECKED_DIR = os.path.join(script_dir, "static/unchecked")
os.makedirs(UNCHECKED_DIR, exist_ok=True)  # Actual upload path for uploaded file

OCR_UPLOAD = os.path.join(script_dir, 'uploads')
os.makedirs(OCR_UPLOAD, exist_ok=True)  # Temporary upload path for scanned pdf file