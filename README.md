# privateGPT
Ask questions to your documents without an internet connection, using the power of LLMs. 100% private, no data leaves your execution environment at any point. You can ingest documents and ask questions without an internet connection!

Built with [LangChain](https://github.com/hwchase17/langchain) and [GPT4All](https://github.com/nomic-ai/gpt4all)

# Environment Setup

In order to set your environment up to run the code here, first install all requirements:

```shell
pip install -r requirements.txt
```

Then, download the 2 models and place them in a folder called `./models`:
- LLM: default to [ggml-gpt4all-j-v1.3-groovy.bin](https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin). If you prefer a different GPT4All-J compatible model, just download it and reference it in `privateGPT.py`.
- Embedding: default to [ggml-model-q4_0.bin](https://huggingface.co/Pi3141/alpaca-native-7B-ggml/resolve/397e872bf4c83f4c642317a5bf65ce84a105786e/ggml-model-q4_0.bin). If you prefer a different compatible Embeddings model, just download it and reference it in `privateGPT.py` and `ingest.py`.

## Test dataset
This repo uses a [state of the union transcript](https://github.com/imartinez/privateGPT/blob/main/source_documents/state_of_the_union.txt) as an example.

## Instructions for ingesting your own dataset

Place your .txt file in `source_documents` folder.
Edit `ingest.py` loader to point it to your document.

Run the following command to ingest the data.

```shell
python ingest.py
```

It will create a `db` folder containing the local vectorstore. Will take time, depending on the size of your document.
You can ingest as many documents as you want by running `ingest`, and all will be accumulated in the local embeddings database. 
If you want to start from scracth, delete the `db` folder.

Note: during the ingest process no data leaves your local environment. You could ingest without an internet connection.

## Ask questions to your documents, locally!
In order to ask a question, run a command like:

```shell
python privateGPT.py
```

And wait for the script to require your input. 

```shell
> Enter a query:
```

Hit enter. You'll see the LLM print the context it is using from your documents and then the final answer; you can then ask another question without re-running the script, just wait for the prompt again. 

Note: you could turn off your internet connection, and the script inference would still work. No data gets out of your local environment.

Type `exit` to finish the script.

# How does it work?
Selecting the right local models and the power of `LangChain` you can run the entire pipeline locally, without any data leaving your environment, and with reasonable performance.

- `ingest.py` uses `LangChain` tools to parse the document and create embeddings locally using `LlamaCppEmbeddings`. It then stores the result in a local vector database using `Chroma` vector store. 
- `privateGPT.py` uses a local LLM based on `GPT4All` to understand questions and create answers. The context for the answers is extracted from the local vector store using a similarity search to locate the right piece of context from the docs.
- `gpt4all_j.py` is a wrapper to support `GPT4All-J` models within LangChain. It was created given such support didn't exist at the moment of creation of this project (only `GPT4All` models where supported). It will be proposed as a contribution to the official `LangChain` repo soon.

# Disclaimer
This is a test project to validate the feasibility of a fully private solution for question answering using LLMs and Vector embeddings. It is not production ready, and it is not meant to be used in production. The models selection is not optimized for performance, but for privacy; but it is possible to use different models and vectorstores to improve performance.