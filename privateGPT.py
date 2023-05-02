from gpt4all_j import GPT4All_J
from langchain.chains import RetrievalQA
from langchain.embeddings import LlamaCppEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma

def main():        
    # Load stored vectorstore
    llama = LlamaCppEmbeddings(model_path="./models/ggml-model-q4_0.bin")
    persist_directory = 'db'
    db = Chroma(persist_directory=persist_directory, embedding_function=llama)
    retriever = db.as_retriever()
    # Prepare the LLM
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = GPT4All_J(model='./models/ggml-gpt4all-j-v1.3-groovy.bin', callbacks=callbacks, verbose=False)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    # Interactive questions and answers
    while True:
        query = input("Enter a query: ")
        if query == "exit":
            break
        qa.run(query)    

if __name__ == "__main__":
    main()