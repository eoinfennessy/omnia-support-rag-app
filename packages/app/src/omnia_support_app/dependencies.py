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
from omnia_support_vectorstore import ChromaVectorStore

from .session import InMemorySessionStore

if TYPE_CHECKING:
    from omnia_support_rag.rag import CompletionsClient
    from omnia_support_vectorstore import VectorStore

    from .session import SessionStore


@dataclass(frozen=True)
class AppConfig:
    """Application configuration loaded from environment."""

    prompt_uri: str
    reformulate_prompt_uri: str
    n_results: int
    llm_model: str


@dataclass
class AppState:
    """Shared application state created once at startup."""

    config: AppConfig
    store: VectorStore
    completions_client: CompletionsClient
    dataset: Dataset
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

    store = ChromaVectorStore(
        host=os.environ.get("CHROMA_HOST", "localhost"),
        port=int(os.environ.get("CHROMA_PORT", "8000")),
        embedding_fn=embedding_fn,  # ty: ignore[invalid-argument-type]
    )

    prompt_name = os.environ.get("PROMPT_NAME", "omnia_support_system")
    prompt_alias = os.environ.get("PROMPT_ALIAS", "production")
    reformulate_prompt_name = os.environ.get("REFORMULATE_PROMPT_NAME", "reformulate_query")

    config = AppConfig(
        prompt_uri=f"prompts:/{prompt_name}@{prompt_alias}",
        reformulate_prompt_uri=f"prompts:/{reformulate_prompt_name}@{prompt_alias}",
        n_results=int(os.environ.get("N_RESULTS", "5")),
        llm_model=os.environ.get("LLM_MODEL", "gpt-4.1-mini"),
    )

    dataset_path = Path(
        os.environ.get("DATASET_PATH", "dataset/omnia_retail_knowledge_base.json"),
    )
    dataset = Dataset.from_json(dataset_path)

    return AppState(
        config=config,
        store=store,
        completions_client=openai.OpenAI().chat.completions,
        dataset=dataset,
    )
