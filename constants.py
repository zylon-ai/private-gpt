import os
from dotenv import load_dotenv
from chromadb.config import Settings


def update_persist_directory():
    load_dotenv(override=True)
    return os.environ.get('PERSIST_DIRECTORY')

def get_chroma_settings():
    # Define the Chroma settings
    return Settings(
            chroma_db_impl='duckdb+parquet',
            persist_directory=update_persist_directory(),
            anonymized_telemetry=False
    )
