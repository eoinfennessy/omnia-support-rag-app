"""Core evaluation orchestration: config, predict function, and MLflow run."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlflow
from mlflow.genai.scorers import (
    Correctness,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    RetrievalSufficiency,
)

from omnia_support_rag import CompletionsClient, ContextProcessor, Rescorer, run_rag_pipeline

from .gold_set import GoldSetEntry, content_hash, sync_dataset

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from openai.types.chat import ChatCompletionMessageParam

    from omnia_support_rag import Retriever
    from omnia_support_rag.query_interpretation import QueryInterpreter

from .scorers import retrieval_mrr, retrieval_recall


@dataclass(frozen=True)
class RunEvaluationConfig:
    """Configuration for an evaluation run."""

    gold_set: list[GoldSetEntry]
    prompt_name: str
    prompt_version: int | None
    prompt_alias: str
    retriever: Retriever
    rescorers: Sequence[Rescorer]
    context_processors: Sequence[ContextProcessor]
    collection_name: str
    embedding_model: str
    completions_client: CompletionsClient
    n_results: int
    llm_model: str
    experiment_name: str
    query_interpreter: QueryInterpreter
    retrieval_mode: str
    cosine_threshold: float
    bm25_threshold: float
    semantic_weight: float
    bm25_weight: float
    use_metadata_filter: bool
    interpret_query: bool
    interpret_query_prompt_uri: str | None


def make_predict_fn(
    *,
    completions_client: CompletionsClient,
    retriever: Retriever,
    prompt_uri: str,
    n_results: int,
    model: str,
    rescorers: Sequence[Rescorer] | None = None,
    context_processors: Sequence[ContextProcessor] | None = None,
    query_interpreter: QueryInterpreter,
    use_metadata_filter: bool = True,
) -> Callable[[str], str]:
    """Create a predict function for mlflow.genai.evaluate()."""

    def predict(question: str, history: list[ChatCompletionMessageParam] | None = None) -> str:
        result = run_rag_pipeline(
            question,
            completions_client=completions_client,
            retriever=retriever,
            prompt_uri=prompt_uri,
            n_results=n_results,
            model=model,
            history=history or None,
            rescorers=rescorers,
            context_processors=context_processors,
            query_interpreter=query_interpreter,
            use_metadata_filter=use_metadata_filter,
        )
        return result.answer

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

    iq_prompt_version = None
    if config.interpret_query and config.interpret_query_prompt_uri:
        iq_prompt = mlflow.genai.load_prompt(config.interpret_query_prompt_uri)
        iq_prompt_version = iq_prompt.version
        print(f"Loaded interpret_query prompt version {iq_prompt_version}")

    print(f"Connected to vector store '{config.collection_name}'")

    print(f"Loaded {len(config.gold_set)} gold set entries")

    gold_set_hash = content_hash(
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
        context_processors=config.context_processors,
        query_interpreter=config.query_interpreter,
        use_metadata_filter=config.use_metadata_filter,
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
                "retrieval_mode": config.retrieval_mode,
                "cosine_threshold": config.cosine_threshold,
                "bm25_threshold": config.bm25_threshold,
                "semantic_weight": config.semantic_weight,
                "bm25_weight": config.bm25_weight,
                "interpret_query": config.interpret_query,
                "interpret_query_prompt_version": iq_prompt_version,
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
                retrieval_recall,
                retrieval_mrr,
            ],
        )

    print(f"\nEvaluation complete. Results at {mlflow.get_tracking_uri()}")
    print(f"  Metrics: {eval_result.metrics}")
