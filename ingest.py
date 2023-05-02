from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings

def main():
    # Load document and split in chunks
    loader = TextLoader('./source_documents/state_of_the_union.txt', encoding='utf8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # Create embeddings
    llama = LlamaCppEmbeddings(model_path="./models/ggml-model-q4_0.bin")
    # Create and store locally vectorstore
    persist_directory = 'db'
    db = Chroma.from_documents(texts, llama, persist_directory=persist_directory)
    db.persist()
    db = None

if __name__ == "__main__":
    main()