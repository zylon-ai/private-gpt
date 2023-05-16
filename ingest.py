import os
import glob
from typing import List
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS


load_dotenv()


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf8")
    elif file_path.endswith(".pdf"):
        loader = PDFMinerLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    return loader.load()[0]


def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    txt_files = glob.glob(os.path.join(source_dir, "**/*.txt"), recursive=True)
    pdf_files = glob.glob(os.path.join(source_dir, "**/*.pdf"), recursive=True)
    csv_files = glob.glob(os.path.join(source_dir, "**/*.csv"), recursive=True)
    all_files = txt_files + pdf_files + csv_files
    return [load_single_document(file_path) for file_path in all_files]


def main():
    # Load environment variables
    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
    embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')

    # Load documents and split in chunks
    print(f"Loading documents from {source_directory}")
    chunk_size = 500
    chunk_overlap = 50
    documents = load_documents(source_directory)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents from {source_directory}")
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} characters each)")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    
    # Create and store locally vectorstore
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None


if __name__ == "__main__":
    main()
