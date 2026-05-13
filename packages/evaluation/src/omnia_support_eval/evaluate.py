"""Evaluate the RAG pipeline against a gold set using MLflow GenAI evaluation.

Runs each gold set question through the full RAG pipeline (retrieve -> prompt -> generate),
then uses MLflow's LLM-as-judge scorers to assess retrieval quality, groundedness,
and correctness. All parameters and metrics are logged to MLflow for comparison.

Usage:
    evaluate [--gold-set PATH] [--prompt-alias ALIAS] [--n-results N] ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import openai
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import create_dataset, get_dataset
from mlflow.genai.scorers import (
    Correctness,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    RetrievalSufficiency,
    scorer,
)

from omnia_support_models import Dataset, TestQuery
from omnia_support_rag import (
    BM25Retriever,
    CompletionsClient,
    ContextFormatter,
    HybridRetriever,
    LLMQueryInterpreter,
    LostInMiddleFormatter,
    Rescorer,
    ResolutionStatusRescorer,
    RetrieverWeight,
    SatisfactionRescorer,
    SemanticRetriever,
    TokenBudgetFormatter,
    run_rag_pipeline,
)
from omnia_support_store import BM25sKeywordStore, ChromaVectorStore

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mlflow.genai.datasets import EvaluationDataset

    from omnia_support_rag import Retriever
    from omnia_support_rag.query_interpretation import QueryInterpreter

DATASET_NAME = "omnia-support-gold-set"


@dataclass(frozen=True)
class RunEvaluationConfig:
    """Configuration for an evaluation run."""

    gold_set: list[GoldSetEntry]
    prompt_name: str
    prompt_version: int | None
    prompt_alias: str
    retriever: Retriever
    rescorers: Sequence[Rescorer]
    context_formatters: Sequence[ContextFormatter]
    collection_name: str
    embedding_model: str
    completions_client: CompletionsClient
    n_results: int
    llm_model: str
    experiment_name: str
    query_interpreter: QueryInterpreter | None = None


class GoldSetEntry(TestQuery):
    """A test query with its expected answer for evaluation."""

    expected_answer: str

    @classmethod
    def from_test_query(cls, query: TestQuery, expected_answer: str) -> GoldSetEntry:
        """Create a gold set entry from a test query and expected answer."""
        return cls(**query.model_dump(), expected_answer=expected_answer)


def load_gold_set(gold_set_path: Path, dataset_path: Path) -> list[GoldSetEntry]:
    """Load and merge gold set entries from the dataset and gold set file.

    Merges expected_answers into test queries from the dataset,
    and appends any additional_queries defined in the gold set.
    """
    if not gold_set_path.exists():
        raise SystemExit(f"Error: gold set file not found: {gold_set_path}")

    gold_data = json.loads(gold_set_path.read_text(encoding="utf-8"))
    expected_answers: dict[str, str] = gold_data.get("expected_answers", {})
    additional_queries: list[dict[str, Any]] = gold_data.get("additional_queries", [])

    dataset = Dataset.from_json(dataset_path)

    entries: list[GoldSetEntry] = [
        GoldSetEntry.from_test_query(query, expected_answers[query.query_id])
        for query in dataset.test_queries
        if query.query_id in expected_answers
    ]
    entries.extend(GoldSetEntry.model_validate(aq) for aq in additional_queries)

    if not entries:
        raise SystemExit("Error: gold set produced no entries")

    return entries


def _content_hash(raw: str) -> str:
    """Compute a SHA-256 hash of the gold set file contents."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sync_dataset(
    entries: list[GoldSetEntry],
    content_hash: str,
    experiment_id: str,
) -> EvaluationDataset:
    """Sync gold set entries into an MLflow dataset, skipping if unchanged."""
    short_hash = content_hash[:12]
    dataset_name = f"{DATASET_NAME}-{short_hash}"

    try:
        dataset = get_dataset(name=dataset_name)
    except MlflowException as e:
        if e.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
    else:
        print(f"  MLflow dataset '{dataset_name}' is up to date")
        return dataset

    print(f"  Creating MLflow dataset '{dataset_name}'...")
    dataset = create_dataset(
        name=dataset_name,
        experiment_id=experiment_id,
    )

    records = [
        {
            "inputs": {"question": entry.query},
            "expectations": {
                "expected_response": entry.expected_answer,
                "expected_document_ids": entry.expected_document_ids,
            },
            "tags": {
                "query_id": entry.query_id,
                "category": entry.category,
            },
        }
        for entry in entries
    ]
    dataset.merge_records(records)
    print(f"  Synced {len(records)} records to MLflow dataset")
    return dataset


@scorer
def retrieval_precision(expectations: dict, trace: Any) -> float | None:
    """Fraction of expected document IDs found in retrieved results.

    Handles chunked IDs like "policy_001-chunk-0" by extracting the
    base document_id from the retrieved record metadata.
    """
    expected = expectations.get("expected_document_ids", [])
    if not expected:
        return None

    retriever_spans = trace.search_spans(span_type="RETRIEVER")
    if not retriever_spans:
        return 0.0

    span = retriever_spans[-1]
    retrieved_doc_ids: set[str] = set()
    if span.outputs and isinstance(span.outputs, list):
        for record in span.outputs:
            if isinstance(record, dict) and "document_id" in record.get("metadata", {}):
                retrieved_doc_ids.add(record["metadata"]["document_id"])

    found = sum(1 for e in expected if e in retrieved_doc_ids)
    return found / len(expected)


def make_predict_fn(
    *,
    completions_client: CompletionsClient,
    retriever: Retriever,
    prompt_uri: str,
    n_results: int,
    model: str,
    rescorers: Sequence[Rescorer] | None = None,
    context_formatters: Sequence[ContextFormatter] | None = None,
    query_interpreter: QueryInterpreter | None = None,
) -> Callable[[str], str]:
    """Create a predict function for mlflow.genai.evaluate()."""

    def predict(question: str) -> str:
        result = run_rag_pipeline(
            question,
            completions_client=completions_client,
            retriever=retriever,
            prompt_uri=prompt_uri,
            n_results=n_results,
            model=model,
            rescorers=rescorers,
            context_formatters=context_formatters,
            query_interpreter=query_interpreter,
        )
        return result["answer"]

    return predict


def run_evaluation(config: RunEvaluationConfig) -> None:
    """Run the evaluation pipeline."""
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
    mlflow.set_experiment(config.experiment_name)

    if config.prompt_version is not None:
        prompt_uri = f"prompts:/{config.prompt_name}/{config.prompt_version}"
    else:
        prompt_uri = f"prompts:/{config.prompt_name}@{config.prompt_alias}"
    prompt = mlflow.genai.load_prompt(prompt_uri)
    print(f"Loaded prompt '{config.prompt_name}' version {prompt.version}")

    print(f"Connected to vector store '{config.collection_name}'")

    print(f"Loaded {len(config.gold_set)} gold set entries")

    gold_set_hash = _content_hash(
        json.dumps([e.model_dump() for e in config.gold_set], sort_keys=True),
    )
    experiment = mlflow.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        raise SystemExit(f"Error: experiment '{config.experiment_name}' not found")
    dataset = sync_dataset(config.gold_set, gold_set_hash, experiment.experiment_id)

    predict_fn = make_predict_fn(
        completions_client=config.completions_client,
        retriever=config.retriever,
        prompt_uri=prompt_uri,
        n_results=config.n_results,
        model=config.llm_model,
        rescorers=config.rescorers,
        context_formatters=config.context_formatters,
        query_interpreter=config.query_interpreter,
    )

    with mlflow.start_run():
        mlflow.log_params(
            {
                "prompt_name": config.prompt_name,
                "prompt_version": prompt.version,
                "embedding_model": config.embedding_model,
                "collection": config.collection_name,
                "n_results": config.n_results,
                "model": config.llm_model,
                "gold_set_size": len(config.gold_set),
            }
        )

        eval_result = mlflow.genai.evaluate(
            predict_fn=predict_fn,
            data=dataset,
            scorers=[
                Correctness(),
                RelevanceToQuery(),
                RetrievalGroundedness(),
                RetrievalRelevance(),
                RetrievalSufficiency(),
                retrieval_precision,
            ],
        )

    print(f"\nEvaluation complete. Results at {mlflow.get_tracking_uri()}")
    print(f"  Metrics: {eval_result.metrics}")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline against a gold set",
    )
    parser.add_argument(
        "--gold-set",
        default="packages/evaluation/gold_set.json",
        help="Path to gold set JSON file (default: packages/evaluation/gold_set.json)",
    )
    parser.add_argument(
        "--dataset",
        default="dataset/omnia_retail_knowledge_base.json",
        help="Path to dataset JSON file (default: dataset/omnia_retail_knowledge_base.json)",
    )
    parser.add_argument(
        "--prompt-name",
        default="omnia_support_system",
        help="MLflow prompt registry name (default: omnia_support_system)",
    )

    version_group = parser.add_mutually_exclusive_group()
    version_group.add_argument(
        "--prompt-version",
        type=int,
        default=None,
        help="Specific prompt version number (default: use --prompt-alias)",
    )
    version_group.add_argument(
        "--prompt-alias",
        default="production",
        help="Prompt alias to load (default: production)",
    )

    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model name (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query (default: 5)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-mini",
        help="LLM model for generation (default: gpt-5.4-mini)",
    )
    parser.add_argument(
        "--chroma-host",
        default=os.environ.get("CHROMA_HOST", "localhost"),
        help="ChromaDB host (default: $CHROMA_HOST or localhost)",
    )
    parser.add_argument(
        "--chroma-port",
        type=int,
        default=int(os.environ.get("CHROMA_PORT", "8000")),
        help="ChromaDB port (default: $CHROMA_PORT or 8000)",
    )
    parser.add_argument(
        "--experiment-name",
        default="omnia-support-eval",
        help="MLflow experiment name (default: omnia-support-eval)",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["semantic", "bm25", "hybrid"],
        default="semantic",
        help="Retrieval strategy: semantic (default), bm25, or hybrid (RRF dense+BM25)",
    )
    parser.add_argument(
        "--bm25-path",
        default=os.environ.get("BM25_INDEX_PATH", "data/bm25"),
        help="Path to BM25 keyword index (default: $BM25_INDEX_PATH or data/bm25)",
    )
    parser.add_argument(
        "--interpret-query-prompt-name",
        default="interpret_query",
        help="MLflow prompt name for query interpretation (default: interpret_query)",
    )
    parser.add_argument(
        "--interpret-query-prompt-alias",
        default="production",
        help="MLflow prompt alias for query interpretation (default: production)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for running evaluation."""
    load_dotenv()
    args = _parse_args()

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

    rescorers: list[Rescorer] = [SatisfactionRescorer(), ResolutionStatusRescorer()]
    context_formatters: list[ContextFormatter] = [TokenBudgetFormatter(), LostInMiddleFormatter()]

    retriever: Retriever
    if args.retrieval_mode == "hybrid":
        if vector_store is None or keyword_store is None:
            raise SystemExit("Error: hybrid mode requires both vector and keyword stores")
        semantic = SemanticRetriever(vector_store)
        keyword = BM25Retriever(keyword_store)
        retriever = HybridRetriever(
            RetrieverWeight(semantic, 1.0),
            RetrieverWeight(keyword, 1.0),
        )
    elif args.retrieval_mode == "bm25":
        if keyword_store is None:
            raise SystemExit("Error: bm25 mode requires a keyword store")
        retriever = BM25Retriever(keyword_store)
    else:
        if vector_store is None:
            raise SystemExit("Error: semantic mode requires a vector store")
        retriever = SemanticRetriever(vector_store)

    gold_set = load_gold_set(Path(args.gold_set), Path(args.dataset))

    query_interpreter = LLMQueryInterpreter(
        completions_client=openai.OpenAI().chat.completions,
        model=args.model,
        prompt_uri=f"prompts:/{args.interpret_query_prompt_name}@{args.interpret_query_prompt_alias}",
    )

    config = RunEvaluationConfig(
        gold_set=gold_set,
        prompt_name=args.prompt_name,
        prompt_version=args.prompt_version,
        prompt_alias=args.prompt_alias,
        retriever=retriever,
        rescorers=rescorers,
        context_formatters=context_formatters,
        collection_name=vector_store.collection_name if vector_store else "n/a",
        embedding_model=args.embedding_model,
        completions_client=openai.OpenAI().chat.completions,
        n_results=args.n_results,
        llm_model=args.model,
        experiment_name=args.experiment_name,
        query_interpreter=query_interpreter,
    )
    run_evaluation(config)
