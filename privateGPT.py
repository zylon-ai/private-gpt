#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp, OpenAI
import os
import argparse

from typing import Any

from langchain.prompts import PromptTemplate

params = {
    'translator': 'GoogleTranslator', # GoogleTranslator or OneRingTranslator.
    'custom_url': "http://127.0.0.1:4990/", # custom url for OneRingTranslator server
    'user_lang': 'ru', # user language two-letters code like "fr", "es" etc.
    'translate_user_input': True, # translate user input to EN
    'translate_system_output': True, # translate system output to UserLang
}
def translator_main(string,from_lang:str,to_lang:str) -> str:
    if from_lang == to_lang: return string

    from deep_translator import GoogleTranslator
    res = ""
    if params['translator'] == "GoogleTranslator":
        res = GoogleTranslator(source=from_lang, target=to_lang).translate(string)
    if params['translator'] == "OneRingTranslator":
        #print("GoogleTranslator using")
        #return GoogleTranslator(source=params['language string'], target='en').translate(string)
        custom_url = params['custom_url']
        if custom_url == "":
            res = "Please, setup custom_url for OneRingTranslator (usually http://127.0.0.1:4990/)"
        else:
            import requests
            response_orig = requests.get(f"{custom_url}translate", params={"text":string,"from_lang":from_lang,"to_lang":to_lang})
            if response_orig.status_code == 200:
                response = response_orig.json()
                #print("OneRingTranslator result:",response)

                if response.get("error") is not None:
                    print(response)
                    res = "ERROR: "+response.get("error")
                elif response.get("result") is not None:
                    res = response.get("result")
                else:
                    print(response)
                    res = "Unknown result from OneRingTranslator"
            elif response_orig.status_code == 404:
                res = "404 error: can't find endpoint"
            elif response_orig.status_code == 500:
                res = "500 error: OneRingTranslator server error"
            else:
                res = f"{response_orig.status_code} error"

    return res

class PromptTemplateTrans(PromptTemplate):
    def format(self, **kwargs: Any) -> str:
        res = super().format(**kwargs)
        if params["translate_user_input"]:
            res = translator_main(res,params["user_lang"],"en")
            print("Translated prompt: ",res)
        return res

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# создаем шаблон для промта
prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Answer the question using the context. 
Context:
{context}
Question: 
{question}
### Response: """

PROMPT = PromptTemplateTrans(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}


load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')

from constants import CHROMA_SETTINGS

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        case "OpenAILocal":
            llm = OpenAI(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False, openai_api_key="nnn")
            import openai
            openai.api_base = "http://127.0.0.1:5001/v1"
        case _default:
            print(f"Model {model_type} not supported!")
            exit;
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source, chain_type_kwargs=chain_type_kwargs)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        if params["translate_system_output"]:
            answer = translator_main(answer,"en",params["user_lang"])

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
