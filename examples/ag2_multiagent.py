"""AG2 Multi-Agent Chat over PrivateGPT.

Demonstrates how AG2 agents can use PrivateGPT's OpenAI-compatible API
to collaborate on answering questions over private documents.

PrivateGPT runs as a local server with document ingestion and RAG.
AG2 agents connect to it as an OpenAI-compatible endpoint, gaining
access to private document context without any data leaving your machine.

Architecture:
    User -> AG2 GroupChat -> PrivateGPT API (local)
    - Researcher: queries with use_context=true for document-grounded answers
    - Analyst: examines and cross-references findings
    - Writer: synthesizes a final comprehensive answer

Prerequisites:
    1. Start PrivateGPT server: PGPT_PROFILES=openai make run
    2. Ingest documents via the PrivateGPT UI or API
    3. Set environment variables (see below)
    4. Run this script: python examples/ag2_multiagent.py

Requires:
    pip install "ag2[openai]>=0.11.4,<1.0"
"""

import json
import os

import requests
from autogen import (
    AssistantAgent,
    GroupChat,
    GroupChatManager,
    LLMConfig,
    UserProxyAgent,
)

# --- Configuration ---

# PrivateGPT runs locally on port 8001 by default
PRIVATEGPT_BASE_URL = os.environ.get("PRIVATEGPT_URL", "http://localhost:8001/v1")
PRIVATEGPT_API_KEY = os.environ.get("PRIVATEGPT_API_KEY", "not-required")

# AG2 agents need their own LLM for reasoning (can be OpenAI or any provider).
# The agents REASON with this LLM but RETRIEVE data from PrivateGPT.
AGENT_LLM_CONFIG = LLMConfig(
    {
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "api_key": os.environ["OPENAI_API_KEY"],
        "api_type": "openai",
    }
)


# --- PrivateGPT Query Tool ---


def query_private_documents(query: str, use_context: bool = True) -> str:
    """Query PrivateGPT's chat API with optional document context.

    Args:
        query: The question to ask.
        use_context: If True, PrivateGPT uses ingested documents as context
            (RAG). If False, uses only the LLM's parametric knowledge.

    Returns:
        JSON string with the response and optional sources.
    """
    try:
        response = requests.post(
            f"{PRIVATEGPT_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {PRIVATEGPT_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "messages": [{"role": "user", "content": query}],
                "use_context": use_context,
                "include_sources": True,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        result: dict[str, object] = {
            "answer": data["choices"][0]["message"]["content"],
            "sources": [],
        }

        # Extract source documents if available
        sources = []
        for source in data.get("sources", []):
            doc_metadata = source.get("document", {}).get("doc_metadata", {})
            sources.append(
                {
                    "document": doc_metadata.get("file_name", "unknown"),
                    "text": source.get("text", "")[:200],
                }
            )
        result["sources"] = sources

        return json.dumps(result, indent=2)
    except requests.ConnectionError:
        return json.dumps(
            {
                "error": (
                    "Cannot connect to PrivateGPT. "
                    "Start it with: PGPT_PROFILES=openai make run"
                )
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


# --- AG2 Agents ---

researcher = AssistantAgent(
    name="Researcher",
    system_message=(
        "You are a research specialist. Use the query_private_docs tool to "
        "search the private document collection. Break complex questions into "
        "sub-queries for thorough coverage. Present findings as structured "
        "bullet points with source references. "
        "Always use the tool - do NOT answer from your own knowledge."
    ),
    llm_config=AGENT_LLM_CONFIG,
)

analyst = AssistantAgent(
    name="Analyst",
    system_message=(
        "You are a data analyst. Review the Researcher's findings for gaps "
        "or contradictions. Use the query_private_docs tool to ask follow-up "
        "questions about specific details. Focus on accuracy and completeness. "
        "Always use the tool - do NOT answer from your own knowledge."
    ),
    llm_config=AGENT_LLM_CONFIG,
)

writer = AssistantAgent(
    name="Writer",
    system_message=(
        "You are a technical writer. Synthesize the Researcher's and Analyst's "
        "findings into a clear, well-structured answer. Include source "
        "references. Do NOT use any tools - work only with what the other "
        "agents found. End your response with TERMINATE."
    ),
    llm_config=AGENT_LLM_CONFIG,
)

user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    code_execution_config=False,
)


# --- Register PrivateGPT as a Tool ---


@user_proxy.register_for_execution()
@researcher.register_for_llm(
    description=(
        "Query the private document collection via PrivateGPT. "
        "Set use_context=true to search ingested documents (RAG mode), "
        "or use_context=false for general LLM knowledge. "
        "Returns the answer and source document references."
    )
)
@analyst.register_for_llm(
    description=(
        "Query the private document collection via PrivateGPT. "
        "Set use_context=true to search ingested documents (RAG mode), "
        "or use_context=false for general LLM knowledge. "
        "Returns the answer and source document references."
    )
)
def query_private_docs(query: str, use_context: bool = True) -> str:
    """Query private documents via PrivateGPT."""
    return query_private_documents(query, use_context)


# --- Run Multi-Agent Chat ---


def main() -> None:
    """Run a multi-agent analysis over private documents."""
    question = os.environ.get(
        "QUERY",
        "Summarize the key findings and recommendations from the ingested documents.",
    )

    group_chat = GroupChat(
        agents=[user_proxy, researcher, analyst, writer],
        messages=[],
        max_round=10,
        speaker_selection_method="auto",
    )

    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=AGENT_LLM_CONFIG,
    )

    print(f"Question: {question}")
    print(f"PrivateGPT endpoint: {PRIVATEGPT_BASE_URL}")
    print("=" * 60)

    user_proxy.run(manager, message=question).process()

    print("=" * 60)
    print("Multi-agent analysis complete.")


if __name__ == "__main__":
    main()
