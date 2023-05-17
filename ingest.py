import os
import sys
import time
import itertools
import threading
import glob
import openai
import streamlit as st
from typing import List
from dotenv import load_dotenv
load_dotenv()

from flair import (
    gears,
    logobig,
    logosmall, 
    byline,
) 

from prompts import get_startup_prompt

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
import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

persist_directory = os.getenv("PERSIST_DIRECTORY")
llama_embeddings_model = os.getenv("LLAMA_EMBEDDINGS_MODEL")
model_type = os.getenv("MODEL_TYPE")
model_path = os.getenv("MODEL_PATH")
model_n_ctx = os.getenv("MODEL_N_CTX")
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.docstore.document import Document
from pdfminer.pdfparser import PDFSyntaxError
from constants import CHROMA_SETTINGS


load_dotenv()



LOADER_MAPPING = {
#    ".csv": (CSVLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (UnstructuredEmailLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".ipynb": (TextLoader, {"encoding": "utf8"}),
    ".js": (TextLoader, {"encoding": "utf8"}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".py": (TextLoader, {"encoding": "utf8"}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
#    ".xlsx": (CSVLoader, {"sheet_name": "Sheet1"}), # use the CSV loader but add sheet_name arg for excel file
}


load_dotenv()
import random

def colored_print(text):
    colors = ['\033[31m', '\033[32m', '\033[33m', '\033[34m', '\033[35m', '\033[36m']
    # choose a random color from the list of colors
    color_choice = random.choice(colors)
    # use chosen color to add color to text
    colored_text = color_choice + text + '\033[0m'
    print(colored_text)

colored_print(gears)
colored_print(logobig)
colored_print(byline)
print(f"\n")

def animate(text="💭 Learning"):
    """Create a colorful loading spinner animation."""
    color_codes = ['\033[95m', '\033[94m', '\033[92m', '\033[93m', '\033[91m', '\033[0m']
    while loading:
        for i, char in enumerate(itertools.cycle("|/-\\")):
            color = color_codes[i % len(color_codes)]
            sys.stdout.write("\r" + color + text + "... " + char)
            sys.stdout.flush()
            time.sleep(0.5)
    sys.stdout.write('\033[0m')  # reset to default color
    sys.stdout.write("\r" + " " * len(text + "... ") + "\r")
    sys.stdout.flush()

def load_single_document(file_path: str) -> Document:
    global loading  # make sure to use the global loading variable
    loading = True
    load_dotenv()  # take environment variables from .env.

    persist_directory = os.getenv("PERSIST_DIRECTORY")
    llama_embeddings_model = os.getenv("LLAMA_EMBEDDINGS_MODEL")
    model_type = os.getenv("MODEL_TYPE")
    model_path = os.getenv("MODEL_PATH")
    model_n_ctx = os.getenv("MODEL_N_CTX")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        
        try:
            doc = loader.load()[0]

            print(f"✅ Successfully loaded document from file {file_path}.\033[0m")

            t = threading.Thread(target=animate)
            t.start()
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": get_startup_prompt()},
                    {"role": "user", "content": doc.page_content[:4096]} 
                ]
            )
            docs = loader.load()
            num_docs = len(docs)
            doc_types = set([type(doc).__name__ for doc in docs])

            print(f"✅ Successfully loaded {num_docs} documents from file {file_path}.\n\033[0m\n")
            colored_print(logosmall)
            loading=False
            print("💡 \033[1m\033[0;36mDreamwalk:\033[0m", response['choices'][0]['message']['content'],)
            print("\n")
            return doc
            
        except UnicodeDecodeError:
            print(f"\033[5m⚠️\033[0m", f" Skipping file {file_path} due to UnicodeDecodeError.\033[0m")
            return None
        except TypeError:
            print(f"\033[5m⚠️\033[0m", f" Skipping file {file_path} due to TypeError (possibly invalid document structure).\033[0m")
            return None
        except PDFSyntaxError:
            print(f"\033[5m⚠️\033[0m", f" Skipping file {file_path} due to PDFSyntaxError (no /root).\033[0m")
            return None
        # except PSEOF:
        #     print(f"⚠️ \033[5m*", f"\033[31mError: Unexpected EOF in file {file_path}. The file might be corrupted or not a valid PDF.\033[0m")
        #     return None
        except Exception as e:
            print(f"\033[5m⚠️\033[0m", f" An unexpected error occurred while processing the file {file_path}: {str(e)}\033[0m")
            return None
    else:
        print(f"\033[5m⚠️ \033[0mSkipping file {file_path} due to unsupported file extension '{ext}'\033[0m")
        return None

def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    valid_files = [file_path for file_path in all_files if not os.path.basename(file_path).startswith('~$')]
    print(f"Found {len(valid_files)} valid files.")
    documents = [load_single_document(file_path) for file_path in valid_files]
    # Filter out any None values from the documents list
    return [doc for doc in documents if doc is not None]



def main():
    # Load environment variables
    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
    llama_embeddings_model = os.environ.get('LLAMA_EMBEDDINGS_MODEL')
    model_n_ctx = os.environ.get('MODEL_N_CTX')

    # Load documents and split in chunks
    print(f"🔄 \033[33mLoading documents from {source_directory}.\033[0m")
    documents = load_documents(source_directory)
    if not documents:
        print("\033[31mNo valid documents found. Exiting.\033[0m")
        return
    print(f"📁 Loaded {len(documents)} documents from {source_directory}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    try:
        texts = text_splitter.split_documents(documents)
        print(f"📁 \033[0;32mSplit into {len(texts)} chunks of text (max. 500 tokens each)\033[0m")
    except AttributeError as e:
        print(f"Error splitting documents: {e}\033[0m")
        return

    # Create embeddings
    print("🔄 \033[33mCreating embeddings.")
    llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
    print("🗄️ \033[0;32mEmbeddings created.")

    # Create and store locally vectorstore
    print("🔄 \033[1;33mCreating vectorstore.")
    db = Chroma.from_documents(texts, llama, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    print("🗄️ \033[0;32mVectorstore created.")

    print("🔄 \033[1;33mPersisting vectorstore.")
    db.persist()
    print("🗄️ \033[0;32mVectorstore persisted.")
    db = None
    
    print("")


if __name__ == "__main__":
    main()

