"""Gold set loading and MLflow dataset synchronisation for action agent evaluation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..common import sync_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from mlflow.genai.datasets import EvaluationDataset

DATASET_NAME = "omnia-support-action-gold-set"


class ActionGoldSetEntry(BaseModel):
    """A test query with expected action agent outputs for evaluation."""

    query_id: str
    query: str
    expected_answer: str
    expected_tools: list[str]
    category: str = ""
    notes: str = ""


def load_gold_set(gold_set_path: Path) -> list[ActionGoldSetEntry]:
    """Load action evaluation entries from the gold set file."""
    if not gold_set_path.exists():
        raise SystemExit(f"Error: gold set file not found: {gold_set_path}")

    gold_data = json.loads(gold_set_path.read_text(encoding="utf-8"))
    entries = [ActionGoldSetEntry.model_validate(q) for q in gold_data.get("action_queries", [])]

    if not entries:
        raise SystemExit("Error: no action_queries found in gold set")

    return entries


def build_records(entries: list[ActionGoldSetEntry]) -> list[dict[str, Any]]:
    """Build MLflow dataset records from action gold set entries."""
    return [
        {
            "inputs": {"question": entry.query},
            "expectations": {
                "expected_response": entry.expected_answer,
                "expected_tools": entry.expected_tools,
            },
            "tags": {
                "query_id": entry.query_id,
                "category": entry.category,
            },
        }
        for entry in entries
    ]


def sync_gold_set(
    entries: list[ActionGoldSetEntry],
    hash_value: str,
    experiment_id: str,
) -> EvaluationDataset:
    """Sync action gold set entries into an MLflow dataset."""
    return sync_dataset(
        build_records(entries),
        hash_value,
        experiment_id,
        dataset_name_prefix=DATASET_NAME,
    )
