import os
import glob
from typing import List
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
from privateGPT.constants import CHROMA_SETTINGS

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
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, doc in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.append(doc)
                pbar.update()

    return results

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False

class Ingestor:
    def __init__(self, persist_directory, source_directory, embeddings_model_name, chunk_size, chunk_overlap):
        self.persist_directory = persist_directory
        self.source_directory = source_directory
        self.embeddings_model_name = embeddings_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    def _process_documents(self, ignored_files: List[str] = []) -> List[Document]:
        """
        Load documents and split in chunks
        """
        print(f"Loading documents from {self.source_directory}")
        documents = load_documents(self.source_directory, ignored_files)
        if not documents:
            return None
        print(f"Loaded {len(documents)} new documents from {self.source_directory}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        texts = text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks of text (max. {self.chunk_size} tokens each)")
        return texts

    def ingest(self, source_dir=None):
        if source_dir is None:
            source_dir = self.source_directory
        
        if does_vectorstore_exist(self.persist_directory):
            # Update and store locally vectorstore
            print(f"Appending to existing vectorstore at {self.persist_directory}")
            db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings, client_settings=CHROMA_SETTINGS)
            collection = db.get()
            texts = self._process_documents([metadata['source'] for metadata in collection['metadatas']])
            if texts is None:
                print("No new documents to load")
                return
            print(f"Creating embeddings. May take some minutes...")
            db.add_documents(texts)
        else:
            # Create and store locally vectorstore
            print("Creating new vectorstore")
            texts = self._process_documents()
            if texts is None:
                print("No new documents to load")
                return
            print(f"Creating embeddings. May take some minutes...")
            db = Chroma.from_documents(texts, self.embeddings, persist_directory=self.persist_directory, client_settings=CHROMA_SETTINGS)
        db.persist()
        db = None

        print(f"Ingestion complete!")