#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from constants import CHROMA_SETTINGS
import os
import time

load_dotenv()


class Config:
    def __init__(self):
        self.embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
        self.persist_directory = os.environ.get('PERSIST_DIRECTORY')
        self.model_type = os.environ.get('MODEL_TYPE')
        self.model_path = os.environ.get('MODEL_PATH')
        self.model_n_ctx = os.environ.get('MODEL_N_CTX')
        self.model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
        self.target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))


class PrivateGPTQueryInterface:
    def __init__(self, config: Config):
        self.mute_stream = False
        self.config = config
        self.llm = None
        self._initialize_model()

    def _initialize_model(self):
        if self.config.model_type == "LlamaCpp":
            self.llm = LlamaCpp(model_path=self.config.model_path, n_ctx=self.config.model_n_ctx,
                                n_batch=self.config.model_n_batch,
                                callbacks=self._get_callbacks(), verbose=False)
        elif self.config.model_type == "GPT4All":
            self.llm = GPT4All(model=self.config.model_path, backend='gptj', n_batch=self.config.model_n_batch,
                               callbacks=self._get_callbacks(), verbose=False)
        else:
            raise Exception(
                f"Model type {self.config.model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    def _get_callbacks(self):
        return [StreamingStdOutCallbackHandler()] if not self.mute_stream else []

    def get_answer(self, query, mute_stream=False, hide_source=False):
        embeddings = HuggingFaceEmbeddings(model_name=self.config.embeddings_model_name)
        db = Chroma(persist_directory=self.config.persist_directory, embedding_function=embeddings,
                    client_settings=CHROMA_SETTINGS)
        retriever = db.as_retriever(search_kwargs={"k": self.config.target_source_chunks})

        if query.strip() == "":
            return {"result": "Invalid query", "source_documents": []}

        qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever,
                                         return_source_documents=not hide_source)

        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if hide_source else res['source_documents']
        end = time.time()

        return {
            "question": query,
            "answer": answer,
            "time_taken": round(end - start, 2),
            "source_documents": docs
        }
