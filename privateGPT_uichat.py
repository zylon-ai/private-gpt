from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import gradio as gr

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')

from constants import CHROMA_SETTINGS

def get_answer(qa, query):
    res = qa(query)
    answer, docs = res['result'], res['source_documents']
    return answer, docs

qa = None

def initialize_qa():
    global qa
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()

    callbacks = [StreamingStdOutCallbackHandler()]

    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {model_type} not supported!")
            exit;

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

def chatbot(query):
    answer, docs = get_answer(qa, query)
    response = f"Question: {query}\n\nAnswer: {answer}\n\n"

    for document in docs:
        response += f"{document.metadata['source']}:\n{document.page_content}\n\n"

    return response.strip()

initialize_qa()

iface = gr.Interface(fn=chatbot, inputs="text", outputs="text", title="Chatbot")
iface.launch()
