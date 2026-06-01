"""Gold set loading and MLflow dataset synchronisation for information agent evaluation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import Field

from omnia_support_models import Dataset, TestQuery

from ..common import sync_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from mlflow.genai.datasets import EvaluationDataset

DATASET_NAME = "omnia-support-gold-set"


class GoldSetEntry(TestQuery):
    """A test query with its expected answer for evaluation."""

    expected_answer: str
    history: list[dict[str, str]] = Field(default_factory=list)

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

    for conv in gold_data.get("conversations", []):
        conv_entries: list[GoldSetEntry] = []
        for turn in conv["turns"]:
            history = [
                msg
                for prior in conv_entries
                for msg in (
                    {"role": "user", "content": prior.query},
                    {"role": "assistant", "content": prior.expected_answer},
                )
            ]
            entry = GoldSetEntry(
                query_id=turn["query_id"],
                query=turn["query"],
                expected_answer=turn["expected_answer"],
                expected_document_ids=turn.get("expected_document_ids", []),
                category=turn.get("category", "multi_turn"),
                notes=turn.get("notes", ""),
                history=history,
            )
            conv_entries.append(entry)
        entries.extend(conv_entries)

    if not entries:
        raise SystemExit("Error: gold set produced no entries")

    return entries


def build_records(entries: list[GoldSetEntry]) -> list[dict[str, Any]]:
    """Build MLflow dataset records from gold set entries."""
    return [
        {
            "inputs": {"question": entry.query, "history": entry.history},
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


def sync_gold_set(
    entries: list[GoldSetEntry],
    hash_value: str,
    experiment_id: str,
) -> EvaluationDataset:
    """Sync information gold set entries into an MLflow dataset."""
    return sync_dataset(
        build_records(entries),
        hash_value,
        experiment_id,
        dataset_name_prefix=DATASET_NAME,
    )
