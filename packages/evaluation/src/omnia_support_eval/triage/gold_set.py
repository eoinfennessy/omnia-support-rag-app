"""Gold set loading and MLflow dataset synchronisation for triage agent evaluation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from omnia_support_agents import SubQuery  # noqa: TC001 — used at runtime by Pydantic

from ..common import sync_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from mlflow.genai.datasets import EvaluationDataset

DATASET_NAME = "omnia-support-triage-gold-set"


class TriageGoldSetEntry(BaseModel):
    """A test query with expected triage outputs for evaluation."""

    query_id: str
    query: str
    expected_urgency: str
    expected_sub_queries: list[SubQuery] = Field(min_length=1)
    history: list[dict[str, str]] = Field(default_factory=list)
    category: str = ""
    notes: str = ""


def load_gold_set(gold_set_path: Path) -> list[TriageGoldSetEntry]:
    """Load triage evaluation entries from the gold set file."""
    if not gold_set_path.exists():
        raise SystemExit(f"Error: gold set file not found: {gold_set_path}")

    gold_data = json.loads(gold_set_path.read_text(encoding="utf-8"))
    entries = [TriageGoldSetEntry.model_validate(q) for q in gold_data.get("triage_queries", [])]

    if not entries:
        raise SystemExit("Error: no triage_queries found in gold set")

    return entries


def build_records(entries: list[TriageGoldSetEntry]) -> list[dict[str, Any]]:
    """Build MLflow dataset records from triage gold set entries."""
    return [
        {
            "inputs": {"question": entry.query, "history": entry.history},
            "expectations": {
                "expected_routing": [sq.agent for sq in entry.expected_sub_queries],
                "expected_urgency": entry.expected_urgency,
                "expected_sub_query_count": len(entry.expected_sub_queries),
                "expected_sub_queries": [sq.model_dump() for sq in entry.expected_sub_queries],
            },
            "tags": {
                "query_id": entry.query_id,
                "category": entry.category,
            },
        }
        for entry in entries
    ]


def sync_gold_set(
    entries: list[TriageGoldSetEntry],
    hash_value: str,
    experiment_id: str,
) -> EvaluationDataset:
    """Sync triage gold set entries into an MLflow dataset."""
    return sync_dataset(
        build_records(entries),
        hash_value,
        experiment_id,
        dataset_name_prefix=DATASET_NAME,
    )
