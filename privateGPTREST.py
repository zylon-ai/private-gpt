#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
import pprint

load_dotenv()


from constants import CHROMA_SETTINGS
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST'])
def query():
    answer = {}
    embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
    persist_directory = os.environ.get('PERSIST_DIRECTORY')

    model_type = os.environ.get('MODEL_TYPE')
    model_path = os.environ.get('MODEL_PATH')
    model_n_ctx = os.environ.get('MODEL_N_CTX')


    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    if model_type=="LlamaCpp":
      llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
    elif model_type=="GPT4All":
      llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
    else:
      print(f"Model {model_type} not supported!")
      exit;
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    query = request.json['question']
    res = qa(query)
    answer['answer'] = res['result']
    return jsonify(answer)
    
app.run(debug=True)
