#!/usr/bin/env python3
import os
import glob
from typing import List
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    GitLoader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS


load_dotenv()
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 500
chunk_overlap = 50


def load():
    clone_url = "https://github.com/gradio-app/gradio"

    repo_path = "source_git/gradio-app"
    #loader = GitLoader(repo_path, clone_url=clone_url)
    loader = GitLoader(repo_path)

    # Cargar los documentos del repositorio
    documents = loader.load()

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(documents), desc='Loading new documents', ncols=80) as pbar:
            for document in documents:
                results.extend(document)
                pbar.update()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)

        print(f"Appending to existing vectorstore at {persist_directory}")
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)

    return results

load()
