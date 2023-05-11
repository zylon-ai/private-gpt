import sys
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings

def main():
    # Check if the input file path is provided
    if len(sys.argv) < 2:
        print("Please provide the path to the input file as a command line argument.")
        return

    input_file_path = sys.argv[1]

    try:
        # Load document and split into chunks
        loader = TextLoader(input_file_path, encoding="utf8")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # Create embeddings
        llama = LlamaCppEmbeddings(model_path="./models/ggml-model-q4_0.bin")

        # Create and store vectorstore
        persist_directory = 'db'
        db = Chroma.from_documents(texts, llama, persist_directory=persist_directory)
        db.persist()
        db = None

        print("Vectorstore created and stored successfully.")

    except FileNotFoundError:
        print("File not found. Please check the file path and try again.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
