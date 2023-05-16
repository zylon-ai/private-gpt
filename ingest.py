import os
import glob
from typing import List
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS


load_dotenv()

persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
llama_embeddings_model = os.environ.get('LLAMA_EMBEDDINGS_MODEL')
model_n_ctx = os.environ.get('MODEL_N_CTX')


def save_to_vectorstore(documents: list[Document], embeddings: LlamaCppEmbeddings):
    if documents:
        chroma = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory,
            client_settings=CHROMA_SETTINGS,
        )
        chroma.get()

        # This is automatically done in Chroma.from_documents() but that method
        # overrides the existing collection, so we need to call .add_texts() ourselves
        # after calling .get()
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        chroma.add_texts(texts=texts, metadatas=metadatas)

        chroma.persist()
    else:
        print("No documents were provided to the store!")


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf8")
    elif file_path.endswith(".pdf"):
        loader = PDFMinerLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    return loader.load()[0]


def find_files(source_dir: str) -> list[str]:
    txt_files = glob.glob(os.path.join(source_dir, "**/*.txt"), recursive=True)
    pdf_files = glob.glob(os.path.join(source_dir, "**/*.pdf"), recursive=True)
    csv_files = glob.glob(os.path.join(source_dir, "**/*.csv"), recursive=True)
    return txt_files + pdf_files + csv_files


def delete_files(files: list[str]):
    for file in files:
        try:
            os.remove(file)
        except Exception:
            print(f"Unable to delete file {file!r}. Please delete it manually.")


def load_documents_from_files(files: list[str]) -> List[Document]:
    # Loads all documents from source documents directory
    return [load_single_document(file_path) for file_path in files]


def main():
    # Load documents and split in chunks
    print(f"Loading documents from {source_directory}")
    files = find_files(source_directory)
    documents = load_documents_from_files(files)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents from {source_directory}")
    print(f"Split into {len(texts)} chunks of text (max. 500 tokens each)")

    # Create embeddings
    llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
    
    # Store documents locally and delete the source files
    save_to_vectorstore(texts, llama)
    delete_files(files)


if __name__ == "__main__":
    main()
