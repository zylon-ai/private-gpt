#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from privateGPT.ingestor import Ingestor
from privateGPT.chat import Chat

import click

load_dotenv()

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Ask questions to your documents without an internet connection, using the power of LLMs."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat)

    pass

@cli.command()
def ingest():
    """Ingest your documents."""
    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
    embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
    chunk_size = 500
    chunk_overlap = 50

    ingestor = Ingestor(
        persist_directory=persist_directory,
        source_directory=source_directory,
        embeddings_model_name=embeddings_model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    ingestor.ingest()

    click.echo(f"You can now run privateGPT.py to query your documents")

@cli.command()
@click.option('--mute-stream', '-M', is_flag=True, default=False, help='Use this flag to disable the streaming StdOut callback for LLMs.')
@click.option('--hide-source', '-S', is_flag=True, default=False, help='Use this flag to disable printing of source documents used for answers.')
def chat(mute_stream, hide_source):
    """Query your documents."""
    embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
    persist_directory = os.environ.get('PERSIST_DIRECTORY')

    model_type = os.environ.get('MODEL_TYPE')
    model_path = os.environ.get('MODEL_PATH')
    model_n_ctx = os.environ.get('MODEL_N_CTX')
    target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

    chat = Chat(
        embeddings_model_name=embeddings_model_name,
        persist_directory=persist_directory,
        model_type=model_type,
        model_path=model_path,
        model_n_ctx=model_n_ctx,
        target_source_chunks=target_source_chunks
    )

    chat.chat(mute_stream, hide_source)

if __name__ == '__main__':
    cli()