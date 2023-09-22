#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import chromadb
import os
import argparse
from torch import cuda as torch_cuda
import time

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
is_gpu_enabled = (os.environ.get('IS_GPU_ENABLED', 'False').lower() == 'true')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
verbose = (os.environ.get('VERBOSE', 'False').lower() == 'true')

from constants import CHROMA_SETTINGS

def get_gpu_memory() -> int:
    """
    Returns the amount of free memory in MB for each GPU.
    """
    return int(torch_cuda.mem_get_info()[0]/(1024**2))

def calculate_layer_count() -> int | None:
    """
    Calculates the number of layers that can be used on the GPU.
    """
    if not is_gpu_enabled:
        return None
    if not torch_cuda.is_available():
        print("You have chosen to use a GPU, but it seems your install was for CPU.\n"
              "Please update through the installer.")
        exit(1)
    if (model_n_gpu_layers := os.environ.get('MODEL_N_GPU_LAYERS')) is not None:  
            return model_n_gpu_layers
    else:
        LAYER_SIZE_MB = 120.6 # This is the size of a single layer on VRAM, and is an approximation.
        # The current set value is for 7B models. For other models, this value should be changed.
        LAYERS_TO_REDUCE = 6 # About 700 MB is needed for the LLM to run, so we reduce the layer count by 6 to be safe.
        if (get_gpu_memory()//LAYER_SIZE_MB) - LAYERS_TO_REDUCE > 32:
            return 32
        else:
            return (get_gpu_memory()//LAYER_SIZE_MB-LAYERS_TO_REDUCE)

model_n_gpu_layers = calculate_layer_count()

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings_kwargs = {'device': 'cuda'} if is_gpu_enabled else {}
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs=embeddings_kwargs)
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_ctx=model_n_ctx, n_batch=model_n_batch, f16_kv=True, callbacks=callbacks, verbose=verbose, n_gpu_layers=model_n_gpu_layers)
        case "GPT4All":
            if is_gpu_enabled:
                print("GPU is enabled, but GPT4All does not support GPU acceleration. Please use LlamaCpp instead.")
                exit(1)
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=verbose)
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

        # Print the relevant sources used for the answer
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


if __name__ == "__main__":
    main()
