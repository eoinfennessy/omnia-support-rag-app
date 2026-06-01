"""Triage agent evaluation: config, predict function, and MLflow run."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlflow

from ..common import content_hash, resolve_prompt_uri
from .gold_set import TriageGoldSetEntry, sync_gold_set
from .scorers import decomposition_accuracy, routing_accuracy, urgency_accuracy

if TYPE_CHECKING:
    from collections.abc import Callable

    from openai.types.chat import ChatCompletionMessageParam

    from omnia_support_agents import TriageAgent


@dataclass(frozen=True)
class RunEvaluationConfig:
    """Configuration for a triage agent evaluation run."""

    gold_set: list[TriageGoldSetEntry]
    prompt_name: str
    prompt_version: int | None
    prompt_alias: str
    triage_agent: TriageAgent
    llm_model: str
    experiment_name: str


def make_predict_fn(
    triage_agent: TriageAgent,
) -> Callable[[str, list[ChatCompletionMessageParam] | None], str]:
    """Create a predict function for mlflow.genai.evaluate()."""

    def predict(question: str, history: list[ChatCompletionMessageParam] | None = None) -> str:
        result = triage_agent.triage(question, history or [])
        return result.model_dump_json()

    return predict


def run_evaluation(config: RunEvaluationConfig) -> None:
    """Run the triage agent evaluation pipeline."""
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

    predict_fn = make_predict_fn(config.triage_agent)

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
                routing_accuracy,
                urgency_accuracy,
                decomposition_accuracy,
            ],
        )

    print(f"\nEvaluation complete. Results at {mlflow.get_tracking_uri()}")
    print(f"  Metrics: {eval_result.metrics}")
