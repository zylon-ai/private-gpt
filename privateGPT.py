from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
import os
import torch

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_name_HuggingFaceHub = os.environ.get('MODEL_NAME_HUGGING_FACE_HUB')

from constants import CHROMA_SETTINGS

def main():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    # Prepare the LLM
    callbacks = [StreamingStdOutCallbackHandler()]
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        case "HuggingFace":
            llm = create_HuggingFace_model(model_name_HuggingFaceHub)
        case _default: 
            print(f"Model {model_type} not supported!")
            exit
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        
        # Get the answer from the chain
        res = qa(query)    
        answer, docs = res['result'], res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)
        
        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

            
def create_HuggingFace_model(model_name_HuggingFaceHub):
    model_id = model_name_HuggingFaceHub
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map='auto')
    pipe = pipeline(
        "text2text-generation",
        model=model, 
        tokenizer=tokenizer)
    return HuggingFacePipeline(pipeline=pipe)
    
if __name__ == "__main__":
    main()
