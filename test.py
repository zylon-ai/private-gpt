import os
import glob
import openai  # Add this line

from typing import List, Optional
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.docstore.document import Document
from pdfminer.pdfparser import PDFSyntaxError
from constants import CHROMA_SETTINGS
from dotenv import load_dotenv  # Importing and loading environment variable from .env file
from langchain.chains import RetrievalQA  # Importing a particular Question-Answer Chain 
from langchain.embeddings import LlamaCppEmbeddings  # Embedding model to convert input text into vectors.
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  
from langchain.vectorstores import Chroma  # Vector store that stores text vectors for large-scale retrieval.
from langchain.llms import GPT4All, LlamaCpp  # Lang Models included here are - LlamaCpp and GPT4All

load_dotenv()

# Initialize OpenAI with API key
openai.api_key = os.getenv("OPENAI_API_KEY")




def generate_response(qa: RetrievalQA, prompt: str) -> str:
    """
    Function to generate a response from a model given a prompt.

    Args:
        qa (RetrievalQA): The RetrievalQA instance to generate the response.
        prompt (str): The prompt for the model.

    Returns:
        str: The generated response.
    """
    res = qa(prompt)     # Query the Question Answer Chain for a response with question as input and storing the result
    answer = res['result']  # Extracting the answer from result.

    return answer




def load_single_document(file_path: str, qa: RetrievalQA) -> Optional[Document]:
    """
    Function to load a single document from a file path.
    It checks the extension of the file and uses the appropriate loader to read it. 

    Args: 
        file_path (str): A string representing the file location on the disk.
        qa (RetrievalQA): The RetrievalQA instance to generate the response.

    Returns:
        Optional[Document]: If a document can be loaded, returns an instance of the Document class.
                            If any error occurs, returns None
    """
    try:
        if file_path.endswith(".txt"):
            prompt = f"Loading {file_path} as text..."
            print(prompt)
            loader = TextLoader(file_path, encoding="utf8")
        elif file_path.endswith(".pdf"):
            prompt = f"Loading {file_path} as PDF..."
            print(prompt)
            loader = PDFMinerLoader(file_path)
        elif file_path.endswith(".csv"):
            prompt = f"Loading {file_path} as CSV..."
            print(prompt)
            loader = CSVLoader(file_path)

        response = generate_response(qa, prompt)
        print(f"Response: {response}")

        return loader.load()[0]

    except PDFSyntaxError:
        print(f"Could not parse {file_path} as PDF. Skipping this file.")
        return None
    except Exception as e:
        print(f"Error loading {file_path}. Error: {str(e)}. Skipping this file.")
        return None


def load_documents(source_dir: str) -> List[Document]:
    """
    Function to load all documents from source documents directory
    It reads all files in the directory with extensions (txt, pdf, csv).
    It calls the load_single_document function for each file.

    Args:
        source_dir (str): Directory path containing source documents.

    Returns:
        List[Document]: If successfully loads documents, returns a list of Document objects.
    """
    txt_files = glob.glob(os.path.join(source_dir, "**/*.txt"), recursive=True)
    pdf_files = glob.glob(os.path.join(source_dir, "**/*.pdf"), recursive=True)
    csv_files = glob.glob(os.path.join(source_dir, "**/*.csv"), recursive=True)
    all_files = txt_files + pdf_files + csv_files
    documents = [doc for doc in (load_single_document(file_path) for file_path in all_files) if doc is not None]
    print(f"Loaded {len(documents)} documents.")
    return documents


def main():
    '''
    Main Function.
        1. Load environment variables which are required for creating embeddings.
        2. Load documents from a source directory and split them into chunks.
        3. Create embeddings for the text chunks
        4. Store the embeddings in a local database for easy access later on
    '''
  # Load environment variables
    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')   # If SOURCE_DIRECTORY is not set, default to "source_documents".
    llama_embeddings_model = os.environ.get('LLAMA_EMBEDDINGS_MODEL')    # Load environment variables for GPT4All model
    model_type = os.getenv('MODEL_TYPE')
    model_path = os.getenv('MODEL_PATH')
    model_n_ctx = int(os.getenv('MODEL_N_CTX'))  # Convert this to an integer
    openai_api_key = os.getenv('OPENAI_API_KEY')

    match model_type:
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', verbose=False)

    # In the main() function, after initializing the GPT4All model, instantiate the RetrievalQA instance

    # Load documents and split in chunks
    print(f"Starting to load documents from {source_directory}")
    documents = load_documents(source_directory)
    print(f"Finished loading documents. Now splitting them into chunks.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # Split the text into 500-token chunks with 50 token overlap
    texts = text_splitter.split_documents(documents)
    print(f"Finished splitting. Total documents: {len(documents)}. Total chunks of text: {len(texts)} (max. 500 tokens each)")

    # Create embeddings
    print(f"Starting to create embeddings...")
    llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
    db = Chroma.from_documents(texts, llama, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    retriever = db.as_retriever()   # Creating an instance of the retriever to perform text search for question answering.

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


    # Generate responses and print them
    for text in texts:
        response = generate_response(qa, text)
        print(response)

    db = None   # Set the vectorstore object to None after use to reduce memory footprint
    print(f"Finished creating and storing vectorstore locally. Ingestion process completed successfully.")


if __name__ == "__main__":
    main()  # Execute the main function, only if this is the top-level script (not imported). 
