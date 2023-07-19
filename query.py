#!/usr/bin/env python3

import logging
import os
import sys
from dotenv import load_dotenv

from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import (
    download_loader,
    LangchainEmbedding,
    ListIndex,
    NotionPageReader,
    ServiceContext,
)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
model_path = os.environ.get("MODEL_PATH")
model_n_ctx = os.environ.get("MODEL_N_CTX")
model_n_batch = int(os.environ.get("MODEL_N_BATCH", 8))
target_source_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS", 4))

# get documents
NotionPageReader = download_loader("NotionPageReader")
integration_token = os.getenv("NOTION_INTEGRATION_TOKEN")
reader = NotionPageReader(integration_token=integration_token)
page_ids = [  # testing only one page for now
    "79d4f07bdace41bba2b84002b1a847ba"  # https://www.notion.so/alleycorpnord/How-we-work-79d4f07bdace41bba2b84002b1a847ba
]
documents = reader.load_data(page_ids=page_ids)

# activate/deactivate the streaming StdOut callback for LLMs
callbacks = [StreamingStdOutCallbackHandler()]

# define LLM
llm = GPT4All(
    model=model_path,
    n_ctx=model_n_ctx,
    backend="gptj",
    n_batch=model_n_batch,
    callbacks=callbacks,
    verbose=False,
)
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name=embeddings_model_name)
)
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

# build index
index = ListIndex.from_documents(documents, service_context=service_context)

# # get response from query
# query = "When is payroll?"
# query_engine = index.as_query_engine()
# response = query_engine.query(query)
# print(f"Response: {response}")
