import os
import glob

def ensure_integrity(persist_directory: str, is_caller_ingest: bool) -> None:
    """
    Checks if vectorstore exists, and if it does, if it is valid
    """
    if not os.path.exists(os.path.join(persist_directory, 'index')) and not is_caller_ingest:
        print("No vectorstore found. Please run ingest.py first.")
        exit(1)
    list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
    list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
    if os.path.exists(os.path.join(persist_directory, 'index')) and (not os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) or not os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')) or not len(list_index_files) > 3):
        print(f"Current vectorstore is not valid. Aborting.")
        print(f"If you deleted any files in the '{persist_directory}' folder, please restore them (if possible) and run privateGPT.py again.")
        print("If you want to start from scratch, delete the '{persist_directory}' folder and its contents and run ingest.py again.")
        exit(1)