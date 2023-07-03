# secureGPT
Ask your documents questions using the power of OpenAI's GPT models, without worrying about your data being used to train models.

using Azure Enterprise OpenAI Service prompts (inputs) and completions (outputs), your embeddings, and your training data:
* are NOT available to other customers.
* are NOT available to OpenAI.
* are NOT used to improve OpenAI models.
* are NOT used to improve any Microsoft or 3rd party products or services.
* are NOT used for automatically improving Azure OpenAI models for your use in your resource (The models are stateless, unless you explicitly fine-tune models with your training data).
* Your fine-tuned Azure OpenAI models are available exclusively for your use.
  
The Azure OpenAI Service is fully controlled by Microsoft; Microsoft hosts the OpenAI models in Microsoft’s Azure environment and the Service does NOT interact with any services operated by OpenAI (e.g. ChatGPT, or the OpenAI API).

# Environment Setup
In order to set your environment up to run the code here, first install all requirements:

```shell
pip3 install -r requirements.txt
```

Copy the `example.env` template into `.env` and update the values in angle brackets
```shell
cp example.env .env
```

## Test dataset
This repo uses a [state of the union transcript](https://github.com/imartinez/privateGPT/blob/main/source_documents/state_of_the_union.txt) as an example.

## Instructions for ingesting your own dataset

Put any and all your files into the `source_documents` directory

The supported extensions are:

   - `.csv`: CSV,
   - `.docx`: Word Document,
   - `.doc`: Word Document,
   - `.enex`: EverNote,
   - `.eml`: Email,
   - `.epub`: EPub,
   - `.html`: HTML File,
   - `.md`: Markdown,
   - `.msg`: Outlook Message,
   - `.odt`: Open Document Text,
   - `.pdf`: Portable Document Format (PDF),
   - `.pptx` : PowerPoint Document,
   - `.ppt` : PowerPoint Document,
   - `.txt`: Text file (UTF-8),

Run the following command to ingest all the data.

```shell
python ingest.py
```

Output should look like this:

```shell
Creating new vectorstore
Loading documents from source_documents
Loading new documents: 100%|██████████████████████| 1/1 [00:01<00:00,  1.73s/it]
Loaded 1 new documents from source_documents
Split into 90 chunks of text (max. 500 tokens each)
Creating embeddings. May take some minutes...
Using embedded DuckDB with persistence: data will be stored in: db
Ingestion complete! You can now run secureGPT.py to query your documents
```

It will create a `db` folder containing the local vectorstore. Will take 20-30 seconds per document, depending on the size of the document.
You can ingest as many documents as you want, and all will be accumulated in the local embeddings database.
If you want to start from an empty database, delete the `db` folder.

Note: during the ingest process no data leaves your local environment. You could ingest without an internet connection, except for the first time you run the ingest script, when the embeddings model is downloaded.

## Ask questions to your documents, locally!
In order to ask a question, run a command like:

```shell
python secureGPT.py
```

And wait for the script to require your input.

```plaintext
> Enter a query:
```

Hit enter. It will print the answer and the 4 sources it used as context from your documents; you can then ask another question without re-running the script, just wait for the prompt again.

Type `exit` to finish the script.


### CLI
The script also supports optional command-line arguments to modify its behavior. You can see a full list of these arguments by running the command ```python secureGPT.py --help``` in your terminal.


# How does it work?

- `ingest.py` uses `LangChain` tools to parse the document and create embeddings locally using `HuggingFaceEmbeddings` (`SentenceTransformers`). It then stores the result in a local vector database using `Chroma` vector store.
- `secureGPT.py` uses a OpenAI's enterprise GPT model to understand questions and create answers. The context for the answers is extracted from the local vector store using a similarity search to locate the right piece of context from the docs.

# System Requirements

## Python Version
To use this software, you must have Python 3.10 or later installed. Earlier versions of Python will not compile.
