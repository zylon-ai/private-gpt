import os
from pathlib import Path

PROJECT_ROOT_PATH: Path = (
    Path(os.environ.get("PGPT_PROJECT_ROOT", str(Path(__file__).parents[1])))
    .expanduser()
    .resolve()
)
