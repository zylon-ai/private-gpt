from chromadb.config import Settings

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        anonymized_telemetry=False
)
