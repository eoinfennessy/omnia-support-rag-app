"""FastAPI dependency providers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import openai
from agents import Agent
from agents.mcp import MCPServerStdio, MCPServerStreamableHttp
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

from omnia_support_agents import (
    AgentOrchestrator,
    AggregatorAgent,
    InformationAgent,
    TriageAgent,
)
from omnia_support_models import Dataset
from omnia_support_rag import (
    BM25Retriever,
    BM25ThresholdFilter,
    CosineSimThresholdFilter,
    DocumentTypeRescorer,
    HybridRetriever,
    LLMQueryInterpreter,
    LostInMiddleFormatter,
    ResolutionStatusRescorer,
    RetrieverWeight,
    SatisfactionRescorer,
    SemanticRetriever,
    TokenBudgetFormatter,
)
from omnia_support_store import BM25sKeywordStore, ChromaVectorStore

from .session import InMemorySessionStore

if TYPE_CHECKING:
    from agents.mcp import MCPServer

    from omnia_support_rag.rag import Retriever

    from .session import SessionStore


@dataclass
class AppState:
    """Shared application state created once at startup."""

    dataset: Dataset
    orchestrator: AgentOrchestrator
    mcp_server: MCPServer
    sessions: SessionStore = field(default_factory=InMemorySessionStore)


def create_app_state() -> AppState:
    """Build application state from environment variables."""
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Error: OPENAI_API_KEY environment variable is not set")

    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=embedding_model,
    )

    retrieval_mode = os.environ.get("RETRIEVAL_MODE", "hybrid")

    vector_store = None
    keyword_store = None
    if retrieval_mode in ("semantic", "hybrid"):
        vector_store = ChromaVectorStore(
            host=os.environ.get("CHROMA_HOST", "localhost"),
            port=int(os.environ.get("CHROMA_PORT", "8000")),
            embedding_fn=embedding_fn,  # ty: ignore[invalid-argument-type]
        )
    if retrieval_mode in ("bm25", "hybrid"):
        keyword_store = BM25sKeywordStore(index_path=Path(os.environ.get("BM25_INDEX_PATH", "data/bm25")))

    rescorers = [DocumentTypeRescorer(), SatisfactionRescorer(), ResolutionStatusRescorer()]
    semantic_filters = [CosineSimThresholdFilter(min_similarity=0.23)]
    keyword_filters = [BM25ThresholdFilter()]

    retriever: Retriever
    if retrieval_mode == "hybrid":
        if vector_store is None or keyword_store is None:
            raise SystemExit("Error: hybrid mode requires both vector and keyword stores")
        semantic = SemanticRetriever(vector_store, filters=semantic_filters)
        keyword = BM25Retriever(keyword_store, filters=keyword_filters)
        retriever = HybridRetriever(
            RetrieverWeight(semantic, 0.6),
            RetrieverWeight(keyword, 0.4),
        )
    elif retrieval_mode == "bm25":
        if keyword_store is None:
            raise SystemExit("Error: bm25 mode requires a keyword store")
        retriever = BM25Retriever(keyword_store, filters=keyword_filters)
    else:
        if vector_store is None:
            raise SystemExit("Error: semantic mode requires a vector store")
        retriever = SemanticRetriever(vector_store, filters=semantic_filters)

    context_processors = [TokenBudgetFormatter(), LostInMiddleFormatter()]

    prompt_name = os.environ.get("PROMPT_NAME", "omnia_support_system")
    prompt_alias = os.environ.get("PROMPT_ALIAS", "production")
    llm_model = os.environ.get("LLM_MODEL", "gpt-5.4-mini")

    prompt_uri = f"prompts:/{prompt_name}@{prompt_alias}"
    n_results = int(os.environ.get("N_RESULTS", "8"))

    interpret_query_prompt_name = os.environ.get("INTERPRET_QUERY_PROMPT_NAME", "interpret_query")
    interpret_query_prompt_alias = os.environ.get("INTERPRET_QUERY_PROMPT_ALIAS", "production")

    openai_client = openai.OpenAI()
    completions_client = openai_client.chat.completions

    query_interpreter = LLMQueryInterpreter(
        completions_client=completions_client,
        model=llm_model,
        prompt_uri=f"prompts:/{interpret_query_prompt_name}@{interpret_query_prompt_alias}",
    )

    dataset_path = Path(
        os.environ.get("DATASET_PATH", "dataset/omnia_retail_knowledge_base.json"),
    )
    dataset = Dataset.from_json(dataset_path)

    triage_agent = TriageAgent(
        completions_client=openai_client.beta.chat.completions,
        model=llm_model,
        prompt_uri=f"prompts:/{os.environ.get('TRIAGE_PROMPT_NAME', 'triage')}@{prompt_alias}",
    )

    information_agent = InformationAgent(
        completions_client=completions_client,
        retriever=retriever,
        prompt_uri=prompt_uri,
        n_results=n_results,
        model=llm_model,
        query_interpreter=query_interpreter,
        rescorers=rescorers,
        context_processors=context_processors,
    )

    aggregator_agent = AggregatorAgent(
        completions_client=completions_client,
        model=llm_model,
        prompt_uri=f"prompts:/{os.environ.get('AGGREGATE_PROMPT_NAME', 'aggregate')}@{prompt_alias}",
    )

    mcp_server_url = os.environ.get("MCP_SERVER_URL")
    if mcp_server_url:
        mcp_server = MCPServerStreamableHttp(
            params={"url": mcp_server_url},
            name="omnia-support-tools",
            cache_tools_list=True,
        )
    else:
        mcp_server = MCPServerStdio(
            params={"command": "uv", "args": ["run", "omnia-support-mcp"]},
            name="omnia-support-tools",
            cache_tools_list=True,
        )

    action_prompt = mlflow.genai.load_prompt(
        f"prompts:/{os.environ.get('ACTION_PROMPT_NAME', 'action')}@{prompt_alias}",
    )
    action_agent = Agent(
        name="action",
        instructions=action_prompt.template[0]["content"],
        model=llm_model,
        mcp_servers=[mcp_server],
    )

    orchestrator = AgentOrchestrator(
        triage_agent=triage_agent,
        information_agent=information_agent,
        aggregator_agent=aggregator_agent,
        action_agent=action_agent,
    )

    return AppState(
        dataset=dataset,
        orchestrator=orchestrator,
        mcp_server=mcp_server,
    )
