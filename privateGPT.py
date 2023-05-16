from dotenv import load_dotenv  # Importing and loading environment variable from .env file
from langchain.chains import RetrievalQA  # Importing a particular Question-Answer Chain 
from langchain.embeddings import LlamaCppEmbeddings  # Embedding model to convert input text into vectors.
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  
from langchain.vectorstores import Chroma  # Vector store that stores text vectors for large-scale retrieval.
from langchain.llms import GPT4All, LlamaCpp  # Lang Models included here are - LlamaCpp and GPT4All
import os

load_dotenv()  # Load environment variables

llama_embeddings_model = os.environ.get("LLAMA_EMBEDDINGS_MODEL")       # Get path of llama embeddings model
persist_directory = os.environ.get('PERSIST_DIRECTORY')    # Get the persistence directory to store chains.

model_type = os.environ.get('MODEL_TYPE')         # Get the type of lang chain
model_path = os.environ.get('MODEL_PATH')         # Get the path of the models (specific to the langchain)
model_n_ctx = os.environ.get('MODEL_N_CTX')       # The number of context words. It represents how many word tokens will be considered before and after a target token for the model.

from constants import CHROMA_SETTINGS   # Here, chroma setting contains dictionary related to vectorisation technique i.e the number of bits used in hashing.

def main():
    llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)  # Creating an instance of LlamaCppEmbeddings() class for embedding.
    db = Chroma(persist_directory=persist_directory, embedding_function=llama, client_settings=CHROMA_SETTINGS)  # Storing the vectors (embeddings) in the vector store named Chroma.
    retriever = db.as_retriever()   # Creating an instance of the retriever to perform text search for question answering.

# Prepare the LLM and Handling Callbacks & Error 
    callbacks = [StreamingStdOutCallbackHandler()]  # A callback handler that feeds stdout to every call, allowing streaming output from other classes.
    match model_type:   # Matching lang_model type and executing appropriate steps.
        case "LlamaCpp":    # If lang_model is LlamapCpp,
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)   # create a new Lang Model instance of LlamaCpp taking model_path and other parameters as arguments
        case "GPT4All":     # If lang_model is GPT4All,
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)  # create a new Lang Model instance of GPT4All taking model path and other parameters as arguments
        case _default:      # If no valid option - 
            print(f"Model {model_type} not supported!")         # Print that not supported
            exit;           # And close the program

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)  # Create an instance of Question-Answer Chain based on the available paramters like Lang Model, Retrieval Model etc.

# Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")    # Get the input query 
        if query == "exit":                   # Exit condition when query is 'exit'
            break
        
        res = qa(query)     # Query the Question Answer Chain for a response with question as input and storing the result
        
        answer, docs = res['result'], res['source_documents']        # Extracting the answer and source documents from result.

        print("\n\n> Question:")    # Printing the question
        print(query)
        print("\n> Answer:")
        print(answer)               # Printing the answer.
        
        for document in docs:       # For every document attached to the answer ,
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)    # Print page content of the relevant documents


if __name__ == "__main__":
    main()
