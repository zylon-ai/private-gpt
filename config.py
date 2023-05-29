import argparse
import os
from dotenv import load_dotenv
from chromadb.config import Settings

load_dotenv()


DEFAULT_CHROMA_DB_IMPL = "duckdb+parquet"
DEFAULT_PERSIST_DIRECTORY = "db"
DEFAULT_CHROMA_PERSIST_DIRECTORY = DEFAULT_PERSIST_DIRECTORY
DEFAULT_CHROMA_TELEMETRY = False
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_CHUNK_SIZE = 500
DEFAULT_EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_MODEL_N_CTX = 1000
DEFAULT_MODEL_PATH = "models/ggml-gpt4all-j-v1.3-groovy.bin"
DEFAULT_MODEL_TYPE = "GPT4All"
DEFAULT_SOURCE_DIRECTORY = "source_documents"
DEFAULT_TARGET_SOURCE_CHUNKS = 4
DEFAULT_MUTE_STREAM = False
DEFAULT_HIDE_SOURCE_DOCUMENTS = False

parser = argparse.ArgumentParser(
    description="privateGPT: Ask questions to your documents without an internet connection, using the power of LLMs."
)
parser.add_argument(
    "--hide-source",
    "-S",
    action="store_true",
    type=bool,
    default=DEFAULT_HIDE_SOURCE_DOCUMENTS,
    help="Use this flag to disable printing of source documents used for answers.",
)

parser.add_argument(
    "--mute-stream",
    "-M",
    type=bool,
    default=DEFAULT_MUTE_STREAM,
    action="store_true",
    help="Use this flag to disable the streaming StdOut callback for LLMs.",
)

parser.add_argument("--persist_directory", default="db", help="Persist directory")
parser.add_argument(
    "--source_directory", default=DEFAULT_SOURCE_DIRECTORY, help="Source directory"
)
parser.add_argument(
    "--embeddings_model_name",
    default=DEFAULT_EMBEDDINGS_MODEL_NAME,
    help="Embeddings model name",
)
parser.add_argument(
    "--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size"
)
parser.add_argument(
    "--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap"
)
parser.add_argument(
    "--target_source_chunks",
    type=int,
    default=DEFAULT_TARGET_SOURCE_CHUNKS,
    help="Target source chunks",
)
parser.add_argument(
    "--chroma_db_impl", default=DEFAULT_CHROMA_DB_IMPL, help="Chroma DB implementation"
)
parser.add_argument(
    "--chroma_telemetry",
    type=bool,
    default=DEFAULT_CHROMA_TELEMETRY,
    help="Chroma telemetry",
)
parser.add_argument(
    "--chroma_persist_directory",
    default=DEFAULT_CHROMA_PERSIST_DIRECTORY,
    help="Chroma persist directory",
)
parser.add_argument("--model_type", default=DEFAULT_MODEL_TYPE, help="Model type")
parser.add_argument(
    "--model_path", default=DEFAULT_MODEL_PATH, help="Model path"
)
parser.add_argument("--model_n_ctx", type=int, default=DEFAULT_MODEL_N_CTX, help="Model n_ctx")

args = parser.parse_args()

PERSIST_DIRECTORY = os.environ.get("PERSIST_DIRECTORY", args.persist_directory)
SOURCE_DIRECTORY = os.environ.get("SOURCE_DIRECTORY", args.source_directory)
EMBEDDINGS_MODEL_NAME = os.environ.get(
    "EMBEDDINGS_MODEL_NAME", args.embeddings_model_name
)
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", args.chunk_size))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", args.chunk_overlap))
TARGET_SOURCE_CHUNKS = int(
    os.environ.get("TARGET_SOURCE_CHUNKS", args.target_source_chunks)
)
CHROMA_DB_IMPL = os.environ.get("CHROMA_DB_IMPL", args.chroma_db_impl)
CHROMA_TELEMETRY = os.environ.get("CHROMA_TELEMETRY", args.chroma_telemetry)
CHROMA_PERSIST_DIRECTORY = os.environ.get(
    "CHROMA_PERSIST_DIRECTORY", args.chroma_persist_directory
)
MODEL_TYPE = os.environ.get("MODEL_TYPE", args.model_type)
MODEL_PATH = os.environ.get("MODEL_PATH", args.model_path)
MODEL_N_CTX = os.environ.get("MODEL_N_CTX", args.model_n_ctx)

CHROMA_SETTINGS = Settings(
    chroma_db_impl=CHROMA_DB_IMPL,
    persist_directory=CHROMA_PERSIST_DIRECTORY,
    anonymized_telemetry=CHROMA_TELEMETRY,
)

HIDE_SOURCE_DOCUMENTS = os.environ.get("HIDE_SOURCE_DOCUMENTS", args.hide_source)
MUTE_STREAM = os.environ.get("MUTE_STREAM", args.mute_stream)