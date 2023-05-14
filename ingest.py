import os
import glob
import time
from typing import List

from dotenv import load_dotenv
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS

from utils import get_logger


load_dotenv()

logger = get_logger("ingest")

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
    llama_embeddings_model = os.environ.get('LLAMA_EMBEDDINGS_MODEL')
    model_n_ctx = os.environ.get('MODEL_N_CTX')

    # Load documents and split in chunks
    logger.info("Loading documents from %s", source_directory)
    documents = load_documents(source_directory)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    logger.info("Loaded %s documents from %s", len(documents), source_directory)
    logger.info("Split into %s chunks of text (max. 500 tokens each)", len(texts))

    # Create embeddings
    llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
    
    # Create and store locally vectorstore
    before = time.time()
    db = Chroma.from_documents(texts, llama, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    after = time.time()
    logger.debug("Took %ss (%smin) to create and store in local vectorstore", round(after - before, 2), round((after - before) / 60, 2))
    db.persist()
    db = None


if __name__ == "__main__":
    main()
