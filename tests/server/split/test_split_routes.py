from fastapi.testclient import TestClient

from private_gpt.server.split.split_router import (
    SplitResponse,
    SplitTextBody,
)


def test_embeddings_generation(test_client: TestClient) -> None:
    body = SplitTextBody(
        text="PrivateGPT is a production-ready AI project that allows you to ask questions about "
        "your documents using the power of Large Language Models (LLMs), even in scenarios "
        "without an Internet connection. 100% private, no data leaves your execution "
        "environment at any point.\n\nThe project provides an API offering all the primitives "
        "required to build private, context-aware AI applications. It follows and extends the "
        "OpenAI API standard, and supports both normal and streaming responses.\n\nThe API is "
        "divided into two logical blocks:\n\nHigh-level API, which abstracts all the "
        "complexity of a RAG (Retrieval Augmented Generation) pipeline "
        "implementation:\n\nIngestion of documents: internally managing document parsing, "
        "splitting, metadata extraction, embedding generation and storage.\nChat & "
        "Completions using context from ingested documents: abstracting the retrieval of "
        "context, the prompt engineering and the response generation.\nLow-level API, "
        "which allows advanced users to implement their own complex pipelines:\n\nEmbeddings "
        "generation: based on a piece of text.\nContextual chunks retrieval: given a query, "
        "returns the most relevant chunks of text from the ingested documents. "
        "The relevance is computed using the cosine similarity between the query and the "
        "chunks' embeddings.\nCompletions generation: given a prompt and a context, "
        "generates a completion using the context as the prompt prefix.\n\nThe project "
        "comes with a pre-trained model, but it is also possible to use your own "
        "pre-trained models. The project supports all the models compatible with the "
        "OpenAI API standard, including GPT-3.\n\nThe project is built on top of the "
        "following open-source libraries:\n\nHuggingFace Transformers: for the "
        "implementation of the LLMs.\nFastAPI: for the implementation of the API.\n\n"
        "The project is production-ready, and it is already being used in production "
        "environments. It is also being used for research purposes, and it is the "
        "foundation of the following research projects:\n\n",
        chunk_size=250,
    )
    response = test_client.post("/v1/split", json=body.model_dump())

    assert response.status_code == 200
    split_response = SplitResponse.model_validate(response.json())
    assert len(split_response.data) > 0
    assert len(split_response.data) == 5
