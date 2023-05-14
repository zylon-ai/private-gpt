import os
from dotenv import load_dotenv
from chromadb.config import Settings

load_dotenv()

# Define the folder for storing database
PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY')
if PERSIST_DIRECTORY is None:
    path_to_dir = os.path.join("privateGPT", "db")
    print(f"Creating a directory for storing database... [path:{path_to_dir}]")
    os.mkdir(path_to_dir)
    os.environ["PERSIST_DIRECTORY"] = path_to_dir

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)