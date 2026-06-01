"""Aggregator agent evaluation: config, predict function, and MLflow run."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery

from omnia_support_agents.models import (
    ActionResult,
    InformationResult,
    ToolCall,
    ToolCallResultErr,
    ToolCallResultOk,
)
from omnia_support_store import ScoredChunk

from ..common import content_hash, resolve_prompt_uri
from .gold_set import AggregatorGoldSetEntry, sync_gold_set

if TYPE_CHECKING:
    from collections.abc import Callable

    from omnia_support_agents import AggregatorAgent
    from omnia_support_store.models import ChunkMetadata

_STUB_METADATA: ChunkMetadata = {  # type: ignore[assignment]
    "document_id": "eval",
    "type": "policy",
    "category": "general",
    "date": "2024-01-01",
    "product_category": "general",
    "chunk_index": 0,
    "total_chunks": 1,
}


@dataclass(frozen=True)
class RunEvaluationConfig:
    """Configuration for an aggregator agent evaluation run."""

    gold_set: list[AggregatorGoldSetEntry]
    prompt_name: str
    prompt_version: int | None
    prompt_alias: str
    aggregator_agent: AggregatorAgent
    llm_model: str
    experiment_name: str


def _build_information_result(inp: dict[str, Any]) -> InformationResult:
    """Convert a serialised information input into an InformationResult."""
    context = [
        ScoredChunk(
            id=f"eval-{i}",
            page_content=c["page_content"],
            metadata=_STUB_METADATA,
        )
        for i, c in enumerate(inp.get("context", []))
    ]
    return InformationResult(query=inp["query"], answer=inp["answer"], context=context)


def _build_action_result(inp: dict[str, Any]) -> ActionResult:
    """Convert a serialised action input into an ActionResult."""
    tool_calls = [
        ToolCall(
            tool_name=tc["tool_name"],
            arguments=tc.get("arguments", {}),
            result=(
                ToolCallResultErr(error=tc["result"]["error"])
                if "error" in tc["result"]
                else ToolCallResultOk(data=tc["result"]["data"])
            ),
        )
        for tc in inp.get("tool_calls", [])
    ]
    return ActionResult(query=inp["query"], answer=inp["answer"], tool_calls=tool_calls)


def make_predict_fn(
    aggregator_agent: AggregatorAgent,
) -> Callable[[str, list[dict[str, Any]], list[dict[str, Any]]], str]:
    """Create a predict function for mlflow.genai.evaluate()."""

    def predict(
        question: str,  # noqa: ARG001
        information_inputs: list[dict[str, Any]],
        action_inputs: list[dict[str, Any]],
    ) -> str:
        results: list[InformationResult | ActionResult] = [
            _build_information_result(inp) for inp in information_inputs
        ] + [_build_action_result(inp) for inp in action_inputs]
        return aggregator_agent.aggregate(results)

    return predict


def run_evaluation(config: RunEvaluationConfig) -> None:
    """Run the aggregator agent evaluation pipeline."""
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
    mlflow.set_experiment(config.experiment_name)

    prompt_uri = resolve_prompt_uri(config.prompt_name, config.prompt_version, config.prompt_alias)
    prompt = mlflow.genai.load_prompt(prompt_uri)
    print(f"Loaded prompt '{config.prompt_name}' version {prompt.version}")

    print(f"Loaded {len(config.gold_set)} gold set entries")

    gold_set_hash = content_hash(
        json.dumps([e.model_dump() for e in config.gold_set], sort_keys=True),
    )
    experiment = mlflow.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        raise SystemExit(f"Error: experiment '{config.experiment_name}' not found")
    dataset = sync_gold_set(config.gold_set, gold_set_hash, experiment.experiment_id)

    predict_fn = make_predict_fn(config.aggregator_agent)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "prompt_name": config.prompt_name,
                "prompt_version": prompt.version,
                "model": config.llm_model,
                "gold_set_size": len(config.gold_set),
            }
        )

        eval_result = mlflow.genai.evaluate(
            predict_fn=predict_fn,
            data=dataset,
            scorers=[
                Correctness(model=f"openai:/{config.llm_model}"),
                RelevanceToQuery(model=f"openai:/{config.llm_model}"),
            ],
        )

    print(f"\nEvaluation complete. Results at {mlflow.get_tracking_uri()}")
    print(f"  Metrics: {eval_result.metrics}")
