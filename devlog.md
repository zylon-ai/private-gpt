Step 1:
Clone the repository

Step 2:
Isolate the code from the repository and environment setup

```shell
cd privateGPT
```

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Step 3:
Download the 2 models and place them in a directory of your choice. (i.e.`/Users/yourusername/downloaded_models/`)

- LLM: default to [ggml-gpt4all-j-v1.3-groovy.bin](https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin). If you prefer a different GPT4All-J compatible model, just download it and reference it in your `.env` file.
- Embedding: default to [ggml-model-q4_0.bin](https://huggingface.co/Pi3141/alpaca-native-7B-ggml/resolve/397e872bf4c83f4c642317a5bf65ce84a105786e/ggml-model-q4_0.bin). If you prefer a different compatible Embeddings model, just download it and reference it in your `.env` file.

Rename `example.env` to `.env` and edit the variables appropriately.

```
MODEL_TYPE: supports LlamaCpp or GPT4All
PERSIST_DIRECTORY: is the folder you want your vectorstore in
LLAMA_EMBEDDINGS_MODEL: (absolute) Path to your LlamaCpp supported embeddings model
MODEL_PATH: Path to your GPT4All or LlamaCpp supported LLM
MODEL_N_CTX: Maximum token limit for both embeddings and LLM models
```

Note: because of the way `langchain` loads the `LLAMMA` embeddings, you need to specify the absolute path of your embeddings model binary. This means it will not work if you use a home directory shortcut (eg. `~/` or `$HOME/`).

Instead you need to use the full path to the embeddings model binary. For example, if you downloaded the embeddings model to your Downloads folder, you would use the following path:

```
LLAMA_EMBEDDINGS_MODEL=/Users/yourusername/downloaded_models/ggml-model-q4_0.bin
```

## Test dataset

This repo uses a [state of the union transcript](https://github.com/imartinez/privateGPT/blob/main/source_documents/state_of_the_union.txt) as an example.

## Instructions for ingesting your own dataset

Step 1:
Put any and all of your .txt, .pdf, or .csv files into the `source_documents` directory

Step 2:
Run the following command to ingest all the data.

```shell
python ingest.py
```

It will create a `db` folder containing the local vectorstore. Will take time, depending on the size of your documents.

You can ingest as many documents as you want, and all will be accumulated in the local embeddings database.

If you want to start from an empty database, delete the `db` folder.

Note: during the ingest process no data leaves your local environment. You could ingest without an internet connection. This is possible because the embeddings are stored locally, and the LLM is loaded locally. Embeddings are not sent to the LLM, only the LLM is sent to the embeddings. An embedding is a vector representation of a word, and the LLM is a language model that can answer questions about a document. The LLM is trained on the embeddings, but the embeddings are not trained on the LLM.

## Ask questions to your documents, locally!

In order to ask a question, run a command like:

```shell
python privateGPT.py
```

And wait for the script to require your input.

```shell
> Enter a query:
```

Hit enter. You'll need to wait 20-30 seconds (depending on your machine) while the LLM model consumes the prompt and prepares the answer. Once done, it will print the answer and the 4 sources it used as context from your documents; you can then ask another question without re-running the script, just wait for the prompt again.

Note: you could turn off your internet connection, and the script inference would still work. No data gets out of your local environment. This is achieved by using the `langchain` library, which is a wrapper around the `transformers` library. The `langchain` library is a fork of the `transformers` library, and it is modified to allow for the LLM to be loaded locally.

## How it works
