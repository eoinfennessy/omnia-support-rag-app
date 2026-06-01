"""Gold set loading and MLflow dataset synchronisation for aggregator agent evaluation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, JsonValue

from ..common import sync_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from mlflow.genai.datasets import EvaluationDataset

DATASET_NAME = "omnia-support-aggregator-gold-set"


class ContextChunk(BaseModel):
    """A single document chunk used as retrieval context."""

    page_content: str


class InformationInput(BaseModel):
    """Pre-canned information agent result for aggregator evaluation."""

    query: str
    answer: str
    context: list[ContextChunk] = Field(default_factory=list)


class ToolCallInput(BaseModel):
    """Pre-canned tool call for aggregator evaluation."""

    tool_name: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    result: dict[str, JsonValue]


class ActionInput(BaseModel):
    """Pre-canned action agent result for aggregator evaluation."""

    query: str
    answer: str
    tool_calls: list[ToolCallInput] = Field(default_factory=list)


class AggregatorGoldSetEntry(BaseModel):
    """A test query with pre-canned sub-query results for aggregator evaluation."""

    query_id: str
    query: str
    expected_answer: str
    information_inputs: list[InformationInput]
    action_inputs: list[ActionInput]
    category: str = ""
    notes: str = ""


def load_gold_set(gold_set_path: Path) -> list[AggregatorGoldSetEntry]:
    """Load aggregator evaluation entries from the gold set file."""
    if not gold_set_path.exists():
        raise SystemExit(f"Error: gold set file not found: {gold_set_path}")

    gold_data = json.loads(gold_set_path.read_text(encoding="utf-8"))
    entries = [AggregatorGoldSetEntry.model_validate(q) for q in gold_data.get("aggregator_queries", [])]

    if not entries:
        raise SystemExit("Error: no aggregator_queries found in gold set")

    return entries


def build_records(entries: list[AggregatorGoldSetEntry]) -> list[dict[str, Any]]:
    """Build MLflow dataset records from aggregator gold set entries."""
    return [
        {
            "inputs": {
                "question": entry.query,
                "information_inputs": [inp.model_dump() for inp in entry.information_inputs],
                "action_inputs": [inp.model_dump() for inp in entry.action_inputs],
            },
            "expectations": {
                "expected_response": entry.expected_answer,
            },
            "tags": {
                "query_id": entry.query_id,
                "category": entry.category,
            },
        }
        for entry in entries
    ]


def sync_gold_set(
    entries: list[AggregatorGoldSetEntry],
    hash_value: str,
    experiment_id: str,
) -> EvaluationDataset:
    """Sync aggregator gold set entries into an MLflow dataset."""
    return sync_dataset(
        build_records(entries),
        hash_value,
        experiment_id,
        dataset_name_prefix=DATASET_NAME,
    )
