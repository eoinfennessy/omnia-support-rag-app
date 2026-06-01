"""CLI argument parsing and component construction for information agent evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import openai

if TYPE_CHECKING:
    import argparse
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

from omnia_support_rag import (
    BM25Retriever,
    BM25ThresholdFilter,
    ContextProcessor,
    CosineSimThresholdFilter,
    DocumentTypeRescorer,
    HybridRetriever,
    LLMQueryInterpreter,
    LostInMiddleFormatter,
    PassthroughQueryInterpreter,
    Rescorer,
    ResolutionStatusRescorer,
    RetrieverWeight,
    SatisfactionRescorer,
    SemanticRetriever,
    TokenBudgetFormatter,
)
from omnia_support_store import BM25sKeywordStore, ChromaVectorStore

from ..common import load_yaml_config, resolve_config, resolve_prompt_uri
from .evaluate import RunEvaluationConfig, run_evaluation
from .gold_set import load_gold_set

_PACKAGE_DIR = Path(__file__).parent

_DEFAULTS: dict[str, Any] = {
    "gold_set": str(_PACKAGE_DIR / "gold_set.json"),
    "dataset": "dataset/omnia_retail_knowledge_base.json",
    "prompt_name": "omnia_support_system",
    "prompt_version": None,
    "prompt_alias": "production",
    "embedding_model": "text-embedding-3-small",
    "n_results": 5,
    "model": "gpt-5.4-mini",
    "chroma_host": os.environ.get("CHROMA_HOST", "localhost"),
    "chroma_port": int(os.environ.get("CHROMA_PORT", "8000")),
    "experiment_name": "omnia-support-eval",
    "retrieval_mode": "semantic",
    "bm25_path": os.environ.get("BM25_INDEX_PATH", "data/bm25"),
    "cosine_threshold": 0.23,
    "bm25_threshold": 1.5,
    "semantic_weight": 0.6,
    "bm25_weight": 0.4,
    "interpret_query_prompt_name": "interpret_query",
    "interpret_query_prompt_version": None,
    "interpret_query_prompt_alias": "production",
    "no_interpret_query": False,
    "no_metadata_filter": False,
}


def add_args(parser: argparse.ArgumentParser) -> None:
    """Register information-agent-specific CLI arguments on *parser*."""
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (CLI args override config values)",
    )
    parser.add_argument("--gold-set", help="Path to gold set JSON file")
    parser.add_argument("--dataset", help="Path to dataset JSON file")
    parser.add_argument("--prompt-name", help="MLflow prompt registry name")

    version_group = parser.add_mutually_exclusive_group()
    version_group.add_argument("--prompt-version", type=int, help="Specific prompt version number")
    version_group.add_argument("--prompt-alias", help="Prompt alias to load")

    parser.add_argument("--embedding-model", help="OpenAI embedding model name")
    parser.add_argument("--n-results", type=int, help="Number of chunks to retrieve per query")
    parser.add_argument("--model", help="LLM model for generation")
    parser.add_argument("--chroma-host", help="ChromaDB host")
    parser.add_argument("--chroma-port", type=int, help="ChromaDB port")
    parser.add_argument("--experiment-name", help="MLflow experiment name")
    parser.add_argument(
        "--retrieval-mode",
        choices=["semantic", "bm25", "hybrid"],
        help="Retrieval strategy: semantic, bm25, or hybrid",
    )
    parser.add_argument("--bm25-path", help="Path to BM25 keyword index")
    parser.add_argument("--cosine-threshold", type=float, help="Cosine similarity threshold for semantic filter")
    parser.add_argument("--bm25-threshold", type=float, help="BM25 score threshold for keyword filter")
    parser.add_argument("--semantic-weight", type=float, help="Semantic retriever weight in hybrid RRF fusion")
    parser.add_argument("--bm25-weight", type=float, help="BM25 retriever weight in hybrid RRF fusion")
    parser.add_argument("--interpret-query-prompt-name", help="MLflow prompt name for query interpretation")

    iq_version_group = parser.add_mutually_exclusive_group()
    iq_version_group.add_argument(
        "--interpret-query-prompt-version", type=int, help="Specific interpret query prompt version number"
    )
    iq_version_group.add_argument("--interpret-query-prompt-alias", help="Interpret query prompt alias to load")
    parser.add_argument(
        "--no-interpret-query",
        action="store_true",
        default=None,
        help="Disable LLM query interpretation (use raw query, no metadata filters)",
    )
    parser.add_argument(
        "--no-metadata-filter",
        action="store_true",
        default=None,
        help="Disable metadata filtering while keeping query interpretation",
    )


def _build_retriever(args: argparse.Namespace, vector_store, keyword_store):
    """Build the retriever based on the retrieval mode."""
    from omnia_support_rag import Retriever

    retriever: Retriever
    if args.retrieval_mode == "hybrid":
        if vector_store is None or keyword_store is None:
            raise SystemExit("Error: hybrid mode requires both vector and keyword stores")
        semantic = SemanticRetriever(vector_store, filters=[CosineSimThresholdFilter(args.cosine_threshold)])
        keyword = BM25Retriever(keyword_store, filters=[BM25ThresholdFilter(args.bm25_threshold)])
        retriever = HybridRetriever(
            RetrieverWeight(semantic, args.semantic_weight),
            RetrieverWeight(keyword, args.bm25_weight),
        )
    elif args.retrieval_mode == "bm25":
        if keyword_store is None:
            raise SystemExit("Error: bm25 mode requires a keyword store")
        retriever = BM25Retriever(keyword_store, filters=[BM25ThresholdFilter(args.bm25_threshold)])
    else:
        if vector_store is None:
            raise SystemExit("Error: semantic mode requires a vector store")
        retriever = SemanticRetriever(vector_store, filters=[CosineSimThresholdFilter(args.cosine_threshold)])

    return retriever


def main(args: argparse.Namespace) -> None:
    """Run information agent evaluation from parsed CLI arguments."""
    load_dotenv()

    yaml_config = load_yaml_config(Path(args.config) if args.config else None)
    resolve_config(args, _DEFAULTS, yaml_config)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Error: OPENAI_API_KEY environment variable is not set")

    embedding_fn = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=args.embedding_model,
    )

    vector_store = None
    keyword_store = None
    if args.retrieval_mode in ("semantic", "hybrid"):
        vector_store = ChromaVectorStore(
            host=args.chroma_host,
            port=args.chroma_port,
            embedding_fn=embedding_fn,  # ty: ignore[invalid-argument-type]
        )
    if args.retrieval_mode in ("bm25", "hybrid"):
        keyword_store = BM25sKeywordStore(index_path=Path(args.bm25_path))

    rescorers: list[Rescorer] = [DocumentTypeRescorer(), SatisfactionRescorer(), ResolutionStatusRescorer()]
    context_processors: list[ContextProcessor] = [TokenBudgetFormatter(), LostInMiddleFormatter()]

    retriever = _build_retriever(args, vector_store, keyword_store)

    gold_set = load_gold_set(Path(args.gold_set), Path(args.dataset))

    if args.no_interpret_query:
        query_interpreter = PassthroughQueryInterpreter()
        iq_prompt_uri = None
    else:
        iq_prompt_uri = resolve_prompt_uri(
            args.interpret_query_prompt_name,
            args.interpret_query_prompt_version,
            args.interpret_query_prompt_alias,
        )
        query_interpreter = LLMQueryInterpreter(
            completions_client=openai.OpenAI().chat.completions,
            model=args.model,
            prompt_uri=iq_prompt_uri,
        )

    config = RunEvaluationConfig(
        gold_set=gold_set,
        prompt_name=args.prompt_name,
        prompt_version=args.prompt_version,
        prompt_alias=args.prompt_alias,
        retriever=retriever,
        rescorers=rescorers,
        context_processors=context_processors,
        collection_name=vector_store.collection_name if vector_store else "n/a",
        embedding_model=args.embedding_model,
        completions_client=openai.OpenAI().chat.completions,
        n_results=args.n_results,
        llm_model=args.model,
        experiment_name=args.experiment_name,
        query_interpreter=query_interpreter,
        retrieval_mode=args.retrieval_mode,
        cosine_threshold=args.cosine_threshold,
        bm25_threshold=args.bm25_threshold,
        semantic_weight=args.semantic_weight,
        bm25_weight=args.bm25_weight,
        use_metadata_filter=not args.no_metadata_filter,
        interpret_query=not args.no_interpret_query,
        interpret_query_prompt_uri=iq_prompt_uri,
    )
    run_evaluation(config)
