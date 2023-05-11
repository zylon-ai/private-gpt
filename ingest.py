from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from sys import argv
from pdfminer.high_level import extract_text
from constants import PERSIST_DIRECTORY
from constants import CHROMA_SETTINGS

def is_pdf(path_to_file):
    try:
        extract_text(path_to_file)
        return True
    except:
        return False

def main():
    file_path = argv[1]

    if is_pdf(file_path):
        loader = PyPDFLoader(file_path)
    else:
        # Load document and split in chunks
        loader = TextLoader(file_path, encoding="utf8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    # Create embeddings
    llama = LlamaCppEmbeddings(model_path="./models/ggml-model-q4_0.bin")
    # Create and store locally vectorstore
    db = Chroma.from_documents(texts, llama, persist_directory=PERSIST_DIRECTORY, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

if __name__ == "__main__":
    main()