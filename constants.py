import os
from dotenv import load_dotenv
from chromadb.config import Settings
import json

load_dotenv()

try:
    load_dotenv()
except Exception as e:
    print("Error loading .env file:", str(e))

# PERSIST_DIRECTORY erorr control
PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY')
if not PERSIST_DIRECTORY:
    print("PERSIST_DIRECTORY is not defined in the environment.")

# Define the folder for storing database
PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY')

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)


with open('settings.json') as f:
    settings = json.load(f)

chroma_db_impl = settings.get('chroma_db_impl')
persist_directory = settings.get('persist_directory')
anonymized_telemetry = settings.get('anonymized_telemetry')

CHROMA_SETTINGS = Settings(
    chroma_db_impl=chroma_db_impl,
    persist_directory=persist_directory,
    anonymized_telemetry=anonymized_telemetry
)
