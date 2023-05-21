#!/usr/bin/env python3
import os
import sys
from typing import List, Optional
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS


load_dotenv()


# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 500
chunk_overlap = 50


# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]

    loader_class, loader_args = LOADER_MAPPING[ext]
    loader = loader_class(file_path, **loader_args)
    return loader.load()[0]


def load_documents(
    source_dir: str,
    ignored_files: Optional[List[str]] = None,
    report_processed_files: bool = True,
    report_skipped_files: bool = True,
    report_ignored_files: bool = True,
    use_process_bar: bool = True,
) -> List[Document]:

    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    if ignored_files is None:
        ignored_files : List[str] = []

    filtered_files = []
    extensions = tuple(LOADER_MAPPING.keys())

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(extensions):
                file_path = os.path.join(root, file)
                if file_path not in ignored_files:
                    filtered_files.append(file_path)
                else:
                    if report_ignored_files:
                        print(f"Ignored '{file_path}' (ignore list)")
            else:
                if report_skipped_files:
                    file_path = os.path.join(root, file)
                    print(f"Skipping '{file_path}' - unmatched extension")

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with (tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) if use_process_bar else None) as pbar:
            for i, doc in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.append(doc)
                if use_process_bar:
                    pbar.update()
                if report_processed_files:
                    file_path = filtered_files[i]
                    print(f"Processed '{file_path}'")

    return results

def process_documents(ignored_files: Optional[List[str]] = None) -> List[Document]:
    """
    Load documents and split in chunks
    """

    if ignored_files is None:
        ignored_files : List[str] = []

    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        sys.exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def does_vectorstore_exist(persist_path: str) -> bool:
    """
    Checks if vectorstore exists
    """
    index_path = os.path.join(persist_path, 'index')
    if os.path.exists(index_path):
        chroma_collections_path = os.path.join(persist_path, 'chroma-collections.parquet')
        chroma_embeddings_path = os.path.join(persist_path, 'chroma-embeddings.parquet')
        if os.path.exists(chroma_collections_path) and os.path.exists(chroma_embeddings_path):
            list_index_files = [f for f in os.listdir(index_path) if f.endswith(('.bin', '.pkl'))]
            if len(list_index_files) > 3:
                return True

    return False

def main():
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        print("Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print("Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()

    print("Ingestion complete! You can now run privateGPT.py to query your documents")


if __name__ == "__main__":
    main()
