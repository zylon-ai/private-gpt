import os
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def file_path() -> str:
    return "test.txt"


def create_test_file(file_path: str) -> None:
    with open(file_path, "w") as f:
        f.write("test")


def clear_log_file(log_file_path: str) -> None:
    if Path(log_file_path).exists():
        os.remove(log_file_path)


def read_log_file(log_file_path: str) -> str:
    with open(log_file_path) as f:
        return f.read()


def init_structure(folder: str, file_path: str) -> None:
    clear_log_file(file_path)
    os.makedirs(folder, exist_ok=True)
    create_test_file(f"{folder}/${file_path}")


def test_ingest_one_file_in_allowed_folder(
    file_path: str, test_client: TestClient
) -> None:
    allowed_folder = "local_data/tests/allowed_folder"
    init_structure(allowed_folder, file_path)

    test_env = os.environ.copy()
    test_env["PGPT_PROFILES"] = "test"
    test_env["LOCAL_INGESTION_ENABLED"] = "True"

    result = subprocess.run(
        ["python", "scripts/ingest_folder.py", allowed_folder],
        capture_output=True,
        text=True,
        env=test_env,
    )

    assert result.returncode == 0, f"Script failed with error: {result.stderr}"
    response_after = test_client.get("/v1/ingest/list")

    count_ingest_after = len(response_after.json()["data"])
    assert count_ingest_after > 0, "No documents were ingested"


def test_ingest_disabled(file_path: str) -> None:
    allowed_folder = "local_data/tests/allowed_folder"
    init_structure(allowed_folder, file_path)

    test_env = os.environ.copy()
    test_env["PGPT_PROFILES"] = "test"
    test_env["LOCAL_INGESTION_ENABLED"] = "False"

    result = subprocess.run(
        ["python", "scripts/ingest_folder.py", allowed_folder],
        capture_output=True,
        text=True,
        env=test_env,
    )

    assert result.returncode != 0, f"Script failed with error: {result.stderr}"
