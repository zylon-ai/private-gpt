import argparse
import os
import shutil

from private_gpt.paths import local_data_path
from private_gpt.settings.settings import settings


def wipe() -> None:
    WIPE_MAP = {
        "simple": wipe_simple,  # node store
        "chroma": wipe_chroma,  # vector store
        "postgres": wipe_postgres,  # node, index and vector store
    }
    for dbtype in ("nodestore", "vectorstore"):
        database = getattr(settings(), dbtype).database
        func = WIPE_MAP.get(database)
        if func:
            func(dbtype)
        else:
            print(f"Unable to wipe database '{database}' for '{dbtype}'")


def wipe_file(file: str) -> None:
    if os.path.isfile(file):
        os.remove(file)
        print(f" - Deleted {file}")


def wipe_tree(path: str) -> None:
    if not os.path.exists(path):
        print(f"Warning: Path not found {path}")
        return
    print(f"Wiping {path}...")
    all_files = os.listdir(path)

    files_to_remove = [file for file in all_files if file != ".gitignore"]
    for file_name in files_to_remove:
        file_path = os.path.join(path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            print(f" - Deleted {file_path}")
        except PermissionError:
            print(
                f"PermissionError: Unable to remove {file_path}. It is in use by another process."
            )
            continue


def wipe_simple(dbtype: str) -> None:
    assert dbtype == "nodestore"
    from llama_index.core.storage.docstore.types import (
        DEFAULT_PERSIST_FNAME as DOCSTORE,
    )
    from llama_index.core.storage.index_store.types import (
        DEFAULT_PERSIST_FNAME as INDEXSTORE,
    )

    for store in (DOCSTORE, INDEXSTORE):
        wipe_file(str((local_data_path / store).absolute()))


def wipe_postgres(dbtype: str) -> None:
    try:
        import psycopg2
    except ImportError as e:
        raise ImportError("Postgres dependencies not found") from e

    cur = conn = None
    try:
        tables = {
            "nodestore": ["data_docstore", "data_indexstore"],
            "vectorstore": ["data_embeddings"],
        }[dbtype]
        connection = settings().postgres.model_dump(exclude_none=True)
        schema = connection.pop("schema_name")
        conn = psycopg2.connect(**connection)
        cur = conn.cursor()
        for table in tables:
            sql = f"DROP TABLE IF EXISTS {schema}.{table}"
            cur.execute(sql)
            print(f"Table {schema}.{table} dropped.")
        conn.commit()
    except psycopg2.Error as e:
        print("Error:", e)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def wipe_chroma(dbtype: str):
    assert dbtype == "vectorstore"
    wipe_tree(str((local_data_path / "chroma_db").absolute()))


if __name__ == "__main__":
    commands = {
        "wipe": wipe,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", help="select a mode to run", choices=list(commands.keys())
    )
    args = parser.parse_args()
    commands[args.mode.lower()]()
