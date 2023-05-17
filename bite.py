import os
import glob
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.docstore.document import Document
from prompts import get_snack_prompt
import openai
import gpt4all

from gpt4all import GPT4All

gptj = GPT4All("ggml-gpt4all-j-v1.3-groovy")
messages = [{"role": "user", "content": "Name 3 colors"}]
gptj.chat_completion(messages)

load_dotenv()
BARF_DIRECTORY='./barf'
# MODEL_TYPE=gpt4all
MODEL_PATH='models/ggml-gpt4all-j-v1.3-groovy.bin'
EMBEDDINGS_MODEL_NAME='all-MiniLM-L6-v25'
MODEL_N_CTX=4096
openai.api_key=os.environ.get('OPENAI_API_KEY')

openai.api_key = os.getenv("OPENAI_API_KEY")
llama_embeddings_model = os.getenv("LLAMA_EMBEDDINGS_MODEL")
model_n_ctx = os.getenv("MODEL_N_CTX")
persist_directory = './.barf'

def load_single_document(file_path: str) -> Document:
    loader = TextLoader(file_path, encoding="utf8")
    doc = loader.load()[0]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": get_snack_prompt()},
            {"role": "user", "content": doc.page_content[:4096]} 
        ]
    )
    print("Dreamwalk:", response['choices'][0]['message']['content'])
    return doc

def load_documents(source_dir: str) -> list[Document]:
    all_files = glob.glob(os.path.join(source_dir, "**/*.txt"), recursive=True)
    print(f"Found {len(all_files)} valid files.")
    return [load_single_document(file_path) for file_path in all_files]

def main():
    #&nbsp;Load environment variables
    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    source_directory ='./jar'
    llama_embeddings_model = os.environ.get('LLAMA_EMBEDDINGS_MODEL')
    model_n_ctx = os.environ.get('MODEL_N_CTX')

    # Load documents and split in chunks
    print(f"🔄 \033[33mLoading documents from {source_directory}.\033[0m")
    documents = load_documents(source_directory)
    if not documents:
        print("\033[31mNo valid documents found. Exiting.\033[0m")
        return
    print(f"📁 Loaded {len(documents)} documents from {source_directory}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=50)
    try:
        texts = text_splitter.split_documents(documents)
        texts = [doc.page_content[:4096] for doc in texts]  # extract page_content and limit each text to 4096 characters
        print(f"📁 \033[0;32mSplit into {len(texts)} chunks of text (max. 4000 tokens each)\033[0m")
    except AttributeError as e:
        print(f"Error splitting documents: {e}\033[0m")
        return

    print("Creating embeddings.")
    llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
    print("Embeddings created.")

    print("Creating vectorstore.")
    db = Chroma.from_documents(texts, llama, persist_directory=persist_directory)
    print("Vectorstore created.")

    print("Persisting vectorstore.")
    db.persist()
    print("Vectorstore persisted.")

if __name__ == "__main__":
    main()
