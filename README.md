# privateGPT

Leverage the power of Large Language Models (LLMs) to query your documents without an internet connection. 100% private - your data never leaves your local environment. Built using [LangChain](https://github.com/hwchase17/langchain), [GPT4All](https://github.com/nomic-ai/gpt4all) and [LlamaCpp](https://github.com/ggerganov/llama.cpp).

<img width="902" alt="demo" src="https://user-images.githubusercontent.com/721666/236942256-985801c9-25b9-48ef-80be-3acbb4575164.png">

# Setup

1. Install dependencies:

   ```shell
   pip install -r requirements.txt
   ```

2. Setup model files:

   - For default setup with `curl`, run:
     ```shell
     cp example.env .env
     mkdir models
     cd models
     curl https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin --output ggml-gpt4all-j-v1.3-groovy.bin
     url="https://huggingface.co/Pi3141/alpaca-native-7B-ggml/resolve/397e872bf4c83f4c642317a5bf65ce84a105786e/ggml-model-q4_0.bin"; curl -L $url -o $(basename $url)
     ```
   - Alternatively, download and reference models in `.env`:
     - LLM: [ggml-gpt4all-j-v1.3-groovy.bin](https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin)
     - Embedding: [ggml-model-q4_0.bin](https://huggingface.co/Pi3141/alpaca-native-7B-ggml/resolve/397e872bf4c83f4c642317a5bf65ce84a105786e/ggml-model-q4_0.bin)

3. Rename `example.env` to `.env` and configure variables accordingly:
   ```
   MODEL_TYPE: LlamaCpp or GPT4All
   PERSIST_DIRECTORY: Directory for your vectorstore
   LLAMA_EMBEDDINGS_MODEL: Absolute path to your LlamaCpp compatible embeddings model
   MODEL_PATH: Path to your GPT4All or LlamaCpp compatible LLM
   MODEL_N_CTX: Maximum token limit for both embeddings and LLM models
   ```
   For `LLAMA_EMBEDDINGS_MODEL`, use the full path. For instance:
   ```
   LLAMA_EMBEDDINGS_MODEL=/Users/yourusername/downloaded_models/ggml-model-q4_0.bin
   ```

# Usage

## Ingesting your Dataset

1. Place your .txt, .pdf, or .csv files into the `source_documents` directory
2. Run `python ingest.py` to ingest the data.

This process creates a `db` directory containing the local vectorstore. The time it takes depends on your dataset's size.

You can ingest multiple documents, all of which will be accumulated in the local embeddings database. To start afresh, delete the `db` directory.

Note: The ingest process is entirely local. The embeddings are stored locally, and the LLM is loaded locally. No data is sent to the LLM; the LLM is sent to the embeddings.

## Querying your Documents

To query your documents, run:

```shell
python privateGPT.py
```

When prompted, enter
