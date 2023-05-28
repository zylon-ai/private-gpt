#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse

load_dotenv()



from privateGPT.constants import CHROMA_SETTINGS

class Chat:
    def __init__(self, embeddings_model_name, persist_directory, model_type, model_path, model_n_ctx, target_source_chunks):
        self.embeddings_model_name = embeddings_model_name
        self.persist_directory = persist_directory
        self.model_type = model_type
        self.model_path = model_path
        self.model_n_ctx = model_n_ctx
        self.target_source_chunks = target_source_chunks

        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        self.db = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings, client_settings=CHROMA_SETTINGS)
        self.retriever = self.db.as_retriever(search_kwargs={"k": target_source_chunks})

    def chat(self, mute_stream, hide_source):
        # activate/deactivate the streaming StdOut callback for LLMs
        callbacks = [] if mute_stream else [StreamingStdOutCallbackHandler()]

        # Prepare the LLM
        match self.model_type:
            case "LlamaCpp":
                llm = LlamaCpp(model_path=self.model_path, n_ctx=self.model_n_ctx, callbacks=callbacks, verbose=False)
            case "GPT4All":
                llm = GPT4All(model=self.model_path, n_ctx=self.model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
            case _default:
                raise(f"Model {self.model_type} not supported!")

        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=self.retriever, return_source_documents= not hide_source)

        # Interactive questions and answers
        while True:
            query = input("\nEnter a query: ")
            if query == "exit":
                break

            # Get the answer from the chain
            res = qa(query)
            answer, docs = res['result'], [] if hide_source else res['source_documents']

            # Print the result
            print("\n\n> Question:")
            print(query)
            print("\n> Answer:")
            print(answer)

            # Print the relevant sources used for the answer
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
