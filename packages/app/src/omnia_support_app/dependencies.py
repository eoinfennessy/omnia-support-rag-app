"""FastAPI dependency providers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import openai
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

from omnia_support_models import Dataset
from omnia_support_rag import (
    BM25Retriever,
    BM25ThresholdFilter,
    CosineSimThresholdFilter,
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
    from collections.abc import Sequence

    from omnia_support_rag.context_formatting import ContextFormatter
    from omnia_support_rag.query_interpretation import QueryInterpreter
    from omnia_support_rag.rag import CompletionsClient, Rescorer, Retriever

    from .session import SessionStore


@dataclass(frozen=True)
class AppConfig:
    """Application configuration loaded from environment."""

    prompt_uri: str
    n_results: int
    llm_model: str


@dataclass
class AppState:
    """Shared application state created once at startup."""

    config: AppConfig
    retriever: Retriever
    rescorers: Sequence[Rescorer]
    context_formatters: Sequence[ContextFormatter]
    completions_client: CompletionsClient
    dataset: Dataset
    query_interpreter: QueryInterpreter | None = None
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

    retrieval_mode = os.environ.get("RETRIEVAL_MODE", "semantic")

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

    rescorers = [SatisfactionRescorer(), ResolutionStatusRescorer()]
    semantic_filters = [CosineSimThresholdFilter()]
    keyword_filters = [BM25ThresholdFilter()]

    retriever: Retriever
    if retrieval_mode == "hybrid":
        if vector_store is None or keyword_store is None:
            raise SystemExit("Error: hybrid mode requires both vector and keyword stores")
        semantic = SemanticRetriever(vector_store, filters=semantic_filters)
        keyword = BM25Retriever(keyword_store, filters=keyword_filters)
        retriever = HybridRetriever(
            RetrieverWeight(semantic, 1.0),
            RetrieverWeight(keyword, 1.0),
        )
    elif retrieval_mode == "bm25":
        if keyword_store is None:
            raise SystemExit("Error: bm25 mode requires a keyword store")
        retriever = BM25Retriever(keyword_store, filters=keyword_filters)
    else:
        if vector_store is None:
            raise SystemExit("Error: semantic mode requires a vector store")
        retriever = SemanticRetriever(vector_store, filters=semantic_filters)

    context_formatters = [TokenBudgetFormatter(), LostInMiddleFormatter()]

    prompt_name = os.environ.get("PROMPT_NAME", "omnia_support_system")
    prompt_alias = os.environ.get("PROMPT_ALIAS", "production")
    llm_model = os.environ.get("LLM_MODEL", "gpt-5.4-mini")

    config = AppConfig(
        prompt_uri=f"prompts:/{prompt_name}@{prompt_alias}",
        n_results=int(os.environ.get("N_RESULTS", "5")),
        llm_model=llm_model,
    )

    interpret_query_prompt_name = os.environ.get("INTERPRET_QUERY_PROMPT_NAME", "interpret_query")
    interpret_query_prompt_alias = os.environ.get("INTERPRET_QUERY_PROMPT_ALIAS", "production")

    query_interpreter = LLMQueryInterpreter(
        completions_client=openai.OpenAI().chat.completions,
        model=llm_model,
        prompt_uri=f"prompts:/{interpret_query_prompt_name}@{interpret_query_prompt_alias}",
    )

    dataset_path = Path(
        os.environ.get("DATASET_PATH", "dataset/omnia_retail_knowledge_base.json"),
    )
    dataset = Dataset.from_json(dataset_path)

    return AppState(
        config=config,
        retriever=retriever,
        rescorers=rescorers,
        context_formatters=context_formatters,
        completions_client=openai.OpenAI().chat.completions,
        dataset=dataset,
        query_interpreter=query_interpreter,
    )
