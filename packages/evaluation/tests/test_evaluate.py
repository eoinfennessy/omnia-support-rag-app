"""Tests for the evaluation module."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST

from omnia_support_eval.gold_set import (
    DATASET_NAME,
    GoldSetEntry,
    content_hash,
    load_gold_set,
    sync_dataset,
)
from omnia_support_eval.scorers import retrieval_mrr, retrieval_recall
from omnia_support_models import TestQuery

if TYPE_CHECKING:
    from pathlib import Path

_retrieval_recall = retrieval_recall._original_func  # ty: ignore[unresolved-attribute]
_retrieval_mrr = retrieval_mrr._original_func  # ty: ignore[unresolved-attribute]

_MINIMAL_DATASET = {
    "dataset_metadata": {
        "name": "test",
        "version": "0.1",
        "created": "2025-01-01",
        "description": "test dataset",
        "total_documents": 1,
        "document_types": {"support_tickets": 0, "policies": 0, "faqs": 1},
        "product_categories": ["laptops"],
        "geographic_context": "IE",
        "currency": "EUR",
        "regulatory_context": [],
    },
    "documents": [],
    "test_queries": [
        {
            "query_id": "TQ-001",
            "query": "How do I return a product?",
            "expected_document_ids": ["policy-001"],
            "category": "returns",
            "notes": "",
        },
        {
            "query_id": "TQ-002",
            "query": "What warranty do I get?",
            "expected_document_ids": ["policy-002"],
            "category": "warranty",
            "notes": "",
        },
    ],
}

_MINIMAL_CONVERSATION = {
    "conversation_id": "CONV-001",
    "turns": [
        {
            "query_id": "CONV-001-T1",
            "query": "What is the return policy?",
            "expected_answer": "30-day return period.",
            "expected_document_ids": ["policy-001"],
            "category": "returns",
            "notes": "",
        },
        {
            "query_id": "CONV-001-T2",
            "query": "What if it's faulty?",
            "expected_answer": "Full refund for faulty items.",
            "expected_document_ids": ["policy-001", "faq-009"],
            "category": "returns",
            "notes": "",
        },
        {
            "query_id": "CONV-001-T3",
            "query": "How long does the refund take?",
            "expected_answer": "Refunds processed within 14 days.",
            "expected_document_ids": ["policy-001"],
            "category": "returns",
            "notes": "",
        },
    ],
}


def _make_test_query(
    *,
    query_id: str = "TQ-001",
    query: str = "Test question?",
    expected_document_ids: list[str] | None = None,
    category: str = "general",
    notes: str = "",
) -> TestQuery:
    return TestQuery(
        query_id=query_id,
        query=query,
        expected_document_ids=expected_document_ids or [],
        category=category,
        notes=notes,
    )


def _make_trace(outputs: list, *, span_count: int = 1) -> MagicMock:
    spans = []
    for i in range(span_count):
        span = MagicMock()
        span.outputs = outputs if i == span_count - 1 else []
        spans.append(span)
    trace = MagicMock()
    trace.search_spans.return_value = spans
    return trace


class TestGoldSetEntry:
    def test_from_test_query_copies_fields(self):
        query = _make_test_query(query_id="TQ-005", query="Why?")

        entry = GoldSetEntry.from_test_query(query, "expected answer")

        assert entry.query_id == "TQ-005"
        assert entry.query == "Why?"
        assert entry.expected_answer == "expected answer"
        assert entry.category == "general"

    def test_from_test_query_preserves_expected_document_ids(self):
        query = _make_test_query(expected_document_ids=["doc-1", "doc-2"])

        entry = GoldSetEntry.from_test_query(query, "answer")

        assert entry.expected_document_ids == ["doc-1", "doc-2"]


class TestLoadGoldSet:
    def test_merges_expected_answers_with_dataset_queries(self, tmp_path: Path):
        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text(json.dumps(_MINIMAL_DATASET))

        gold_set_path = tmp_path / "gold_set.json"
        gold_set_path.write_text(
            json.dumps(
                {
                    "expected_answers": {"TQ-001": "Return within 14 days."},
                }
            )
        )

        entries = load_gold_set(gold_set_path, dataset_path)

        assert len(entries) == 1
        assert entries[0].query_id == "TQ-001"
        assert entries[0].expected_answer == "Return within 14 days."

    def test_includes_additional_queries(self, tmp_path: Path):
        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text(json.dumps(_MINIMAL_DATASET))

        gold_set_path = tmp_path / "gold_set.json"
        gold_set_path.write_text(
            json.dumps(
                {
                    "expected_answers": {"TQ-001": "answer 1"},
                    "additional_queries": [
                        {
                            "query_id": "AQ-001",
                            "query": "What about quantum?",
                            "expected_answer": "Not available",
                            "expected_document_ids": [],
                            "category": "no_result",
                            "notes": "",
                        },
                    ],
                }
            )
        )

        entries = load_gold_set(gold_set_path, dataset_path)

        assert len(entries) == 2
        query_ids = {e.query_id for e in entries}
        assert query_ids == {"TQ-001", "AQ-001"}

    def test_raises_system_exit_when_gold_set_missing(self, tmp_path: Path):
        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text(json.dumps(_MINIMAL_DATASET))

        with pytest.raises(SystemExit, match="gold set file not found"):
            load_gold_set(tmp_path / "missing.json", dataset_path)

    def test_raises_system_exit_when_no_entries(self, tmp_path: Path):
        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text(json.dumps(_MINIMAL_DATASET))

        gold_set_path = tmp_path / "gold_set.json"
        gold_set_path.write_text(json.dumps({"expected_answers": {}}))

        with pytest.raises(SystemExit, match="no entries"):
            load_gold_set(gold_set_path, dataset_path)


class TestConversationExpansion:
    def test_expands_conversation_turns_into_entries(self, tmp_path: Path):
        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text(json.dumps(_MINIMAL_DATASET))

        gold_set_path = tmp_path / "gold_set.json"
        gold_set_path.write_text(
            json.dumps(
                {
                    "expected_answers": {},
                    "conversations": [_MINIMAL_CONVERSATION],
                }
            )
        )

        entries = load_gold_set(gold_set_path, dataset_path)

        assert len(entries) == 3
        assert [e.query_id for e in entries] == [
            "CONV-001-T1",
            "CONV-001-T2",
            "CONV-001-T3",
        ]

    def test_first_turn_has_empty_history(self, tmp_path: Path):
        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text(json.dumps(_MINIMAL_DATASET))

        gold_set_path = tmp_path / "gold_set.json"
        gold_set_path.write_text(
            json.dumps(
                {
                    "expected_answers": {},
                    "conversations": [_MINIMAL_CONVERSATION],
                }
            )
        )

        entries = load_gold_set(gold_set_path, dataset_path)

        assert entries[0].history == []

    def test_second_turn_has_first_turn_history(self, tmp_path: Path):
        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text(json.dumps(_MINIMAL_DATASET))

        gold_set_path = tmp_path / "gold_set.json"
        gold_set_path.write_text(
            json.dumps(
                {
                    "expected_answers": {},
                    "conversations": [_MINIMAL_CONVERSATION],
                }
            )
        )

        entries = load_gold_set(gold_set_path, dataset_path)

        assert entries[1].history == [
            {"role": "user", "content": "What is the return policy?"},
            {"role": "assistant", "content": "30-day return period."},
        ]

    def test_third_turn_has_two_prior_turns(self, tmp_path: Path):
        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text(json.dumps(_MINIMAL_DATASET))

        gold_set_path = tmp_path / "gold_set.json"
        gold_set_path.write_text(
            json.dumps(
                {
                    "expected_answers": {},
                    "conversations": [_MINIMAL_CONVERSATION],
                }
            )
        )

        entries = load_gold_set(gold_set_path, dataset_path)

        assert entries[2].history == [
            {"role": "user", "content": "What is the return policy?"},
            {"role": "assistant", "content": "30-day return period."},
            {"role": "user", "content": "What if it's faulty?"},
            {"role": "assistant", "content": "Full refund for faulty items."},
        ]

    def test_combined_with_single_turn_entries(self, tmp_path: Path):
        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text(json.dumps(_MINIMAL_DATASET))

        gold_set_path = tmp_path / "gold_set.json"
        gold_set_path.write_text(
            json.dumps(
                {
                    "expected_answers": {"TQ-001": "Return within 14 days."},
                    "conversations": [_MINIMAL_CONVERSATION],
                }
            )
        )

        entries = load_gold_set(gold_set_path, dataset_path)

        assert len(entries) == 4
        assert entries[0].query_id == "TQ-001"
        assert entries[0].history == []

    def test_single_turn_entries_have_empty_history(self, tmp_path: Path):
        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text(json.dumps(_MINIMAL_DATASET))

        gold_set_path = tmp_path / "gold_set.json"
        gold_set_path.write_text(json.dumps({"expected_answers": {"TQ-001": "Answer."}}))

        entries = load_gold_set(gold_set_path, dataset_path)

        assert entries[0].history == []


class TestContentHash:
    def test_returns_sha256_hex_digest(self):
        expected = hashlib.sha256(b"hello").hexdigest()

        assert content_hash("hello") == expected

    def test_different_inputs_produce_different_hashes(self):
        assert content_hash("a") != content_hash("b")

    def test_deterministic(self):
        assert content_hash("test") == content_hash("test")


class TestRetrievalPrecision:
    def test_returns_none_when_no_expected_ids(self):
        trace = _make_trace([])

        assert _retrieval_recall({"expected_document_ids": []}, trace) is None

    def test_returns_none_when_key_missing(self):
        trace = _make_trace([])

        assert _retrieval_recall({}, trace) is None

    def test_returns_zero_when_no_retriever_spans(self):
        trace = MagicMock()
        trace.search_spans.return_value = []

        result = _retrieval_recall({"expected_document_ids": ["doc-1"]}, trace)

        assert result == 0.0

    def test_perfect_precision(self):
        outputs = [
            {"id": "doc-1-chunk-0", "chunk": "text", "metadata": {"document_id": "doc-1"}, "score": 0.9},
            {"id": "doc-2-chunk-0", "chunk": "text", "metadata": {"document_id": "doc-2"}, "score": 0.8},
        ]
        trace = _make_trace(outputs)

        result = _retrieval_recall(
            {"expected_document_ids": ["doc-1", "doc-2"]},
            trace,
        )

        assert result == 1.0

    def test_partial_precision(self):
        outputs = [
            {"id": "doc-1-chunk-0", "chunk": "text", "metadata": {"document_id": "doc-1"}, "score": 0.9},
        ]
        trace = _make_trace(outputs)

        result = _retrieval_recall(
            {"expected_document_ids": ["doc-1", "doc-2"]},
            trace,
        )

        assert result == 0.5

    def test_uses_last_retriever_span(self):
        wrong_span = MagicMock()
        wrong_span.outputs = [
            {"id": "wrong-chunk-0", "chunk": "x", "metadata": {"document_id": "wrong"}, "score": 0.5},
        ]
        right_span = MagicMock()
        right_span.outputs = [
            {"id": "doc-1-chunk-0", "chunk": "x", "metadata": {"document_id": "doc-1"}, "score": 0.9},
        ]
        trace = MagicMock()
        trace.search_spans.return_value = [wrong_span, right_span]

        result = _retrieval_recall(
            {"expected_document_ids": ["doc-1"]},
            trace,
        )

        assert result == 1.0

    def test_handles_non_dict_outputs(self):
        outputs = [
            "not a dict",
            {"id": "doc-1-chunk-0", "chunk": "x", "metadata": {"document_id": "doc-1"}, "score": 0.9},
            42,
        ]
        trace = _make_trace(outputs)

        result = _retrieval_recall(
            {"expected_document_ids": ["doc-1"]},
            trace,
        )

        assert result == 1.0


def _make_mrr_trace(outputs: list, *, use_reorder_span: bool = True) -> MagicMock:
    """Build a trace mock for MRR tests.

    When *use_reorder_span* is True, the outputs appear in a ``reorder_by_score``
    span (the primary lookup).  When False, they fall back to a RETRIEVER span.
    """
    reorder_span = MagicMock()
    reorder_span.outputs = outputs if use_reorder_span else []
    retriever_span = MagicMock()
    retriever_span.outputs = outputs if not use_reorder_span else []
    trace = MagicMock()

    def _search(*, name=None, span_type=None):
        if name == "reorder_by_score":
            return [reorder_span] if use_reorder_span else []
        if span_type == "RETRIEVER":
            return [retriever_span] if not use_reorder_span else []
        return []

    trace.search_spans.side_effect = _search
    return trace


class TestRetrievalMrr:
    def test_returns_none_when_no_expected_ids(self):
        trace = _make_mrr_trace([])

        assert _retrieval_mrr({"expected_document_ids": []}, trace) is None

    def test_returns_none_when_key_missing(self):
        trace = _make_mrr_trace([])

        assert _retrieval_mrr({}, trace) is None

    def test_returns_zero_when_no_spans(self):
        trace = MagicMock()
        trace.search_spans.return_value = []

        assert _retrieval_mrr({"expected_document_ids": ["doc-1"]}, trace) == 0.0

    def test_uses_reorder_span(self):
        outputs = [
            {"id": "doc-1-chunk-0", "chunk": "text", "metadata": {"document_id": "doc-1"}, "score": 0.9},
        ]
        trace = _make_mrr_trace(outputs, use_reorder_span=True)

        assert _retrieval_mrr({"expected_document_ids": ["doc-1"]}, trace) == 1.0

    def test_falls_back_to_retriever_span(self):
        outputs = [
            {"id": "doc-1-chunk-0", "chunk": "text", "metadata": {"document_id": "doc-1"}, "score": 0.9},
        ]
        trace = _make_mrr_trace(outputs, use_reorder_span=False)

        assert _retrieval_mrr({"expected_document_ids": ["doc-1"]}, trace) == 1.0

    def test_second_result_relevant(self):
        outputs = [
            {"id": "doc-x-chunk-0", "chunk": "text", "metadata": {"document_id": "doc-x"}, "score": 0.9},
            {"id": "doc-1-chunk-0", "chunk": "text", "metadata": {"document_id": "doc-1"}, "score": 0.8},
        ]
        trace = _make_mrr_trace(outputs)

        assert _retrieval_mrr({"expected_document_ids": ["doc-1"]}, trace) == 0.5

    def test_third_result_relevant(self):
        outputs = [
            {"id": "doc-x-chunk-0", "chunk": "text", "metadata": {"document_id": "doc-x"}, "score": 0.9},
            {"id": "doc-y-chunk-0", "chunk": "text", "metadata": {"document_id": "doc-y"}, "score": 0.8},
            {"id": "doc-1-chunk-0", "chunk": "text", "metadata": {"document_id": "doc-1"}, "score": 0.7},
        ]
        trace = _make_mrr_trace(outputs)

        result = _retrieval_mrr({"expected_document_ids": ["doc-1"]}, trace)

        assert abs(result - 1.0 / 3) < 1e-9

    def test_no_relevant_results(self):
        outputs = [
            {"id": "doc-x-chunk-0", "chunk": "text", "metadata": {"document_id": "doc-x"}, "score": 0.9},
        ]
        trace = _make_mrr_trace(outputs)

        assert _retrieval_mrr({"expected_document_ids": ["doc-1"]}, trace) == 0.0

    def test_multiple_expected_returns_rank_of_first_found(self):
        outputs = [
            {"id": "doc-x-chunk-0", "chunk": "text", "metadata": {"document_id": "doc-x"}, "score": 0.9},
            {"id": "doc-2-chunk-0", "chunk": "text", "metadata": {"document_id": "doc-2"}, "score": 0.8},
            {"id": "doc-1-chunk-0", "chunk": "text", "metadata": {"document_id": "doc-1"}, "score": 0.7},
        ]
        trace = _make_mrr_trace(outputs)

        assert _retrieval_mrr({"expected_document_ids": ["doc-1", "doc-2"]}, trace) == 0.5


def _make_gold_entry(
    *,
    query_id: str = "TQ-001",
    query: str = "Test?",
    expected_answer: str = "Answer",
    expected_document_ids: list[str] | None = None,
    category: str = "general",
    notes: str = "",
) -> GoldSetEntry:
    return GoldSetEntry(
        query_id=query_id,
        query=query,
        expected_answer=expected_answer,
        expected_document_ids=expected_document_ids or [],
        category=category,
        notes=notes,
    )


@patch("omnia_support_eval.gold_set.create_dataset")
@patch("omnia_support_eval.gold_set.get_dataset")
class TestSyncDataset:
    def test_returns_existing_dataset_when_hash_matches(self, mock_get, mock_create):
        existing = MagicMock()
        mock_get.return_value = existing
        entries = [_make_gold_entry()]

        result = sync_dataset(entries, "abc123def456", "exp-1")

        expected_name = f"{DATASET_NAME}-abc123def456"
        mock_get.assert_called_once_with(name=expected_name)
        mock_create.assert_not_called()
        assert result is existing

    def test_creates_dataset_when_not_found(self, mock_get, mock_create):
        mock_get.side_effect = MlflowException("not found", error_code=RESOURCE_DOES_NOT_EXIST)
        created = MagicMock()
        mock_create.return_value = created
        entries = [_make_gold_entry(query_id="TQ-001", query="Q?", expected_answer="A")]

        result = sync_dataset(entries, "abc123def456", "exp-1")

        expected_name = f"{DATASET_NAME}-abc123def456"
        mock_create.assert_called_once_with(name=expected_name, experiment_id="exp-1")
        created.merge_records.assert_called_once()
        records = created.merge_records.call_args[0][0]
        assert len(records) == 1
        assert records[0]["inputs"]["question"] == "Q?"
        assert records[0]["expectations"]["expected_response"] == "A"
        assert result is created

    def test_reraises_unexpected_mlflow_errors(self, mock_get, mock_create):
        mock_get.side_effect = MlflowException("server error", error_code=INTERNAL_ERROR)

        with pytest.raises(MlflowException, match="server error"):
            sync_dataset([_make_gold_entry()], "abc123def456", "exp-1")

        mock_create.assert_not_called()

    def test_uses_first_12_chars_of_hash_in_name(self, mock_get, mock_create):  # noqa: ARG002
        existing = MagicMock()
        mock_get.return_value = existing
        full_hash = "a" * 64

        sync_dataset([_make_gold_entry()], full_hash, "exp-1")

        expected_name = f"{DATASET_NAME}-aaaaaaaaaaaa"
        mock_get.assert_called_once_with(name=expected_name)

    def test_merges_multiple_entries(self, mock_get, mock_create):
        mock_get.side_effect = MlflowException("not found", error_code=RESOURCE_DOES_NOT_EXIST)
        created = MagicMock()
        mock_create.return_value = created
        entries = [
            _make_gold_entry(query_id="TQ-001", query="Q1?", expected_answer="A1"),
            _make_gold_entry(query_id="TQ-002", query="Q2?", expected_answer="A2", category="returns"),
        ]

        sync_dataset(entries, "abc123def456", "exp-1")

        records = created.merge_records.call_args[0][0]
        assert len(records) == 2
        assert records[0]["tags"]["query_id"] == "TQ-001"
        assert records[1]["tags"]["query_id"] == "TQ-002"
        assert records[1]["tags"]["category"] == "returns"

    def test_includes_history_in_inputs(self, mock_get, mock_create):
        mock_get.side_effect = MlflowException("not found", error_code=RESOURCE_DOES_NOT_EXIST)
        created = MagicMock()
        mock_create.return_value = created
        history = [
            {"role": "user", "content": "Prior question"},
            {"role": "assistant", "content": "Prior answer"},
        ]
        entries = [_make_gold_entry(query_id="TQ-001", query="Follow up?", expected_answer="A")]
        entries[0].history = history

        sync_dataset(entries, "abc123def456", "exp-1")

        records = created.merge_records.call_args[0][0]
        assert records[0]["inputs"]["history"] == history

    def test_single_turn_has_empty_history_in_inputs(self, mock_get, mock_create):
        mock_get.side_effect = MlflowException("not found", error_code=RESOURCE_DOES_NOT_EXIST)
        created = MagicMock()
        mock_create.return_value = created
        entries = [_make_gold_entry()]

        sync_dataset(entries, "abc123def456", "exp-1")

        records = created.merge_records.call_args[0][0]
        assert records[0]["inputs"]["history"] == []
