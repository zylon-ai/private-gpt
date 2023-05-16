import os
import glob
from typing import List, Optional
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.docstore.document import Document
from pdfminer.pdfparser import PDFSyntaxError
from constants import CHROMA_SETTINGS


load_dotenv()


def load_single_document(file_path: str) -> Optional[Document]:
    # Loads a single document from a file path
    try:
        if file_path.endswith(".txt"):
            print(f"Loading {file_path} as text...")
            loader = TextLoader(file_path, encoding="utf8")
        elif file_path.endswith(".pdf"):
            print(f"Loading {file_path} as PDF...")
            loader = PDFMinerLoader(file_path)
        elif file_path.endswith(".csv"):
            print(f"Loading {file_path} as CSV...")
            loader = CSVLoader(file_path)
        return loader.load()[0]
    except PDFSyntaxError:
        print(f"Could not parse {file_path} as PDF. Skipping this file.")
        return None
    except Exception as e:
        print(f"Error loading {file_path}. Error: {str(e)}. Skipping this file.")
        return None


def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    txt_files = glob.glob(os.path.join(source_dir, "**/*.txt"), recursive=True)
    pdf_files = glob.glob(os.path.join(source_dir, "**/*.pdf"), recursive=True)
    csv_files = glob.glob(os.path.join(source_dir, "**/*.csv"), recursive=True)
    all_files = txt_files + pdf_files + csv_files
    documents = [doc for doc in (load_single_document(file_path) for file_path in all_files) if doc is not None]
    print(f"Loaded {len(documents)} documents.")
    return documents


def main():
    # Load environment variables
    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
    llama_embeddings_model = os.environ.get('LLAMA_EMBEDDINGS_MODEL')
    model_n_ctx = os.environ.get('MODEL_N_CTX')

    # Load documents and split in chunks
    print(f"Starting to load documents from {source_directory}")
    documents = load_documents(source_directory)
    print(f"Finished loading documents. Now splitting them into chunks.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Finished splitting. Total documents: {len(documents)}. Total chunks of text: {len(texts)} (max. 500 tokens each)")

    # Create embeddings
    print(f"Starting to create embeddings...")
    llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
    print(f"Finished creating embeddings.")

    # Create and store locally vectorstore
    print(f"Starting to create and store vectorstore locally...")
    db = Chroma.from_documents(texts, llama, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None
    print(f"Finished creating and storing vectorstore locally. Ingestion process completed successfully.")


if __name__ == "__main__":
    main()
