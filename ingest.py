from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from sys import argv
from chroma_preference import PERSIST_DIRECTORY
from chroma_preference import CHROMA_SETTINGS

def main():
    # Load document and split in chunks
    loader = TextLoader(argv[1], encoding="utf8")
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