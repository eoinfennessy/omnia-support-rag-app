"""Custom MLflow GenAI scorers for information agent / RAG evaluation."""

from __future__ import annotations

from typing import Any

from mlflow.genai.scorers import scorer


@scorer
def retrieval_recall(expectations: dict, trace: Any) -> float | None:
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


def _extract_doc_ids(span: Any) -> list[str]:
    """Extract ordered document IDs from a span's outputs."""
    if not span.outputs or not isinstance(span.outputs, list):
        return []
    return [
        record["metadata"]["document_id"]
        for record in span.outputs
        if isinstance(record, dict) and "document_id" in record.get("metadata", {})
    ]


def _get_ranked_doc_ids(trace: Any) -> list[str]:
    """Extract ranked document IDs from the last RETRIEVER span."""
    retriever_spans = trace.search_spans(span_type="RETRIEVER")
    if not retriever_spans:
        return []
    return _extract_doc_ids(retriever_spans[-1])


@scorer
def retrieval_mrr(expectations: dict, trace: Any) -> float | None:
    """Mean Reciprocal Rank: 1/rank of the first relevant document after reranking."""
    expected = set(expectations.get("expected_document_ids", []))
    if not expected:
        return None

    reorder_spans = trace.search_spans(name="reorder_by_score")
    ranked_ids = _extract_doc_ids(reorder_spans[-1]) if reorder_spans else _get_ranked_doc_ids(trace)

    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0
