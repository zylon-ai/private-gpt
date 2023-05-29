#!/usr/bin/env python3
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import config


def main():
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDINGS_MODEL_NAME)
    db = Chroma(
        persist_directory=config.PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=config.CHROMA_SETTINGS,
    )
    retriever = db.as_retriever(search_kwargs={"k": config.TARGET_SOURCE_CHUNKS})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if config.MUTE_STREAM else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match config.MODEL_TYPE:
        case "LlamaCpp":
            llm = LlamaCpp(
                model_path=config.MODEL_PATH,
                n_ctx=config.MODEL_N_CTX,
                callbacks=callbacks,
                verbose=False,
            )
        case "GPT4All":
            llm = GPT4All(
                model=config.MODEL_PATH,
                n_ctx=config.MODEL_N_CTX,
                backend="gptj",
                callbacks=callbacks,
                verbose=False,
            )
        case _default:
            print(f"Model {config.MODEL_TYPE} not supported!")
            exit(1)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=not config.MUTE_STREAM,
    )
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain
        res = qa(query)
        answer, docs = (
            res["result"],
            [] if config.HIDE_SOURCE_DOCUMENTS else res["source_documents"],
        )

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)


if __name__ == "__main__":
    main()
