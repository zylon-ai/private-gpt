#!/usr/bin/env python3
import os
import argparse
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from constants import CHROMA_SETTINGS

load_dotenv()

def run_qa_system(args):
    embeddings = HuggingFaceEmbeddings(model_name=os.environ.get("EMBEDDINGS_MODEL_NAME"))
    db = Chroma(persist_directory=os.environ.get('PERSIST_DIRECTORY'), embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    match args.model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=os.environ.get('MODEL_PATH'), n_ctx=os.environ.get('MODEL_N_CTX'), callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=os.environ.get('MODEL_PATH'), n_ctx=os.environ.get('MODEL_N_CTX'), backend='gptj', callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {args.model_type} not supported!")
            exit()

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not args.hide_source)

    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']

        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')
    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    run_qa_system(args)

if __name__ == "__main__":
    main()
