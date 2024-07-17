import argparse
import os
import shutil
from typing import Any, ClassVar

from private_gpt.paths import local_data_path
from private_gpt.settings.settings import settings


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


class Postgres:
    tables: ClassVar[dict[str, list[str]]] = {
        "nodestore": ["data_docstore", "data_indexstore"],
        "vectorstore": ["data_embeddings"],
    }

    def __init__(self) -> None:
        try:
            import psycopg2
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Postgres dependencies not found") from None

        connection = settings().postgres.model_dump(exclude_none=True)
        self.schema = connection.pop("schema_name")
        self.conn = psycopg2.connect(**connection)

    def wipe(self, storetype: str) -> None:
        cur = self.conn.cursor()
        try:
            for table in self.tables[storetype]:
                sql = f"DROP TABLE IF EXISTS {self.schema}.{table}"
                cur.execute(sql)
                print(f"Table {self.schema}.{table} dropped.")
            self.conn.commit()
        finally:
            cur.close()

    def stats(self, store_type: str) -> None:
        template = "SELECT '{table}', COUNT(*), pg_size_pretty(pg_total_relation_size('{table}')) FROM {table}"
        sql = " UNION ALL ".join(
            template.format(table=tbl) for tbl in self.tables[store_type]
        )

        cur = self.conn.cursor()
        try:
            print(f"Storage for Postgres {store_type}.")
            print("{:<15} | {:>15} | {:>9}".format("Table", "Rows", "Size"))
            print("-" * 45)  # Print a line separator

            cur.execute(sql)
            for row in cur.fetchall():
                formatted_row_count = f"{row[1]:,}"
                print(f"{row[0]:<15} | {formatted_row_count:>15} | {row[2]:>9}")

            print()
        finally:
            cur.close()

    def __del__(self):
        if hasattr(self, "conn") and self.conn:
            self.conn.close()


class Simple:
    def wipe(self, store_type: str) -> None:
        assert store_type == "nodestore"
        from llama_index.core.storage.docstore.types import (
            DEFAULT_PERSIST_FNAME as DOCSTORE,
        )
        from llama_index.core.storage.index_store.types import (
            DEFAULT_PERSIST_FNAME as INDEXSTORE,
        )

        for store in (DOCSTORE, INDEXSTORE):
            wipe_file(str((local_data_path / store).absolute()))


class Chroma:
    def wipe(self, store_type: str) -> None:
        assert store_type == "vectorstore"
        wipe_tree(str((local_data_path / "chroma_db").absolute()))


class Qdrant:
    COLLECTION = (
        "make_this_parameterizable_per_api_call"  # ?! see vector_store_component.py
    )

    def __init__(self) -> None:
        try:
            from qdrant_client import QdrantClient  # type: ignore
        except ImportError:
            raise ImportError("Qdrant dependencies not found") from None
        self.client = QdrantClient(**settings().qdrant.model_dump(exclude_none=True))

    def wipe(self, store_type: str) -> None:
        assert store_type == "vectorstore"
        try:
            self.client.delete_collection(self.COLLECTION)
            print("Collection dropped successfully.")
        except Exception as e:
            print("Error dropping collection:", e)

    def stats(self, store_type: str) -> None:
        print(f"Storage for Qdrant {store_type}.")
        try:
            collection_data = self.client.get_collection(self.COLLECTION)
            if collection_data:
                # Collection Info
                # https://qdrant.tech/documentation/concepts/collections/
                print(f"\tPoints:        {collection_data.points_count:,}")
                print(f"\tVectors:       {collection_data.vectors_count:,}")
                print(f"\tIndex Vectors: {collection_data.indexed_vectors_count:,}")
                return
        except ValueError:
            pass
        print("\t- Qdrant collection not found or empty")


class Command:
    DB_HANDLERS: ClassVar[dict[str, Any]] = {
        "simple": Simple,  # node store
        "chroma": Chroma,  # vector store
        "postgres": Postgres,  # node, index and vector store
        "qdrant": Qdrant,  # vector store
    }

    def for_each_store(self, cmd: str):
        for store_type in ("nodestore", "vectorstore"):
            database = getattr(settings(), store_type).database
            handler_class = self.DB_HANDLERS.get(database)
            if handler_class is None:
                print(f"No handler found for database '{database}'")
                continue
            handler_instance = handler_class()  # Instantiate the class
            # If the DB can handle this cmd dispatch it.
            if hasattr(handler_instance, cmd) and callable(
                func := getattr(handler_instance, cmd)
            ):
                func(store_type)
            else:
                print(
                    f"Unable to execute command '{cmd}' on '{store_type}' in database '{database}'"
                )

    def execute(self, cmd: str) -> None:
        if cmd in ("wipe", "stats"):
            self.for_each_store(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="select a mode to run", choices=["wipe", "stats"])
    args = parser.parse_args()

    Command().execute(args.mode.lower())
