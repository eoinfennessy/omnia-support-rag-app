"""Custom MLflow GenAI scorers for triage agent evaluation."""

from __future__ import annotations

import json

from mlflow.genai.scorers import scorer


def _parse_triage_result(raw: str) -> dict | None:
    """Parse the serialized TriageResult JSON, returning None on failure."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError, TypeError:
        return None


@scorer
def routing_accuracy(outputs: str, expectations: dict) -> float | None:
    """Fraction of routed agent positions matching the expected routing."""
    expected = expectations.get("expected_routing")
    if expected is None:
        return None

    result = _parse_triage_result(outputs)
    if result is None:
        return 0.0

    actual = sorted(sq["agent"] for sq in result.get("sub_queries", []))
    expected_sorted = sorted(expected)

    if not expected_sorted:
        return None

    max_len = max(len(actual), len(expected_sorted))
    matches = sum(a == e for a, e in zip(actual, expected_sorted, strict=False))
    return matches / max_len


@scorer
def urgency_accuracy(outputs: str, expectations: dict) -> float | None:
    """1.0 if predicted urgency matches expected, 0.0 otherwise."""
    expected = expectations.get("expected_urgency")
    if expected is None:
        return None

    result = _parse_triage_result(outputs)
    if result is None:
        return 0.0

    return 1.0 if result.get("urgency") == expected else 0.0


@scorer
def decomposition_accuracy(outputs: str, expectations: dict) -> float | None:
    """How close the sub-query count is to expected (ratio-based)."""
    expected = expectations.get("expected_sub_query_count")
    if expected is None:
        return None

    result = _parse_triage_result(outputs)
    if result is None:
        return 0.0

    actual = len(result.get("sub_queries", []))
    if expected == 0 and actual == 0:
        return 1.0

    return 1.0 - abs(actual - expected) / max(actual, expected)
