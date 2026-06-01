"""Custom MLflow GenAI scorers for action agent evaluation."""

from __future__ import annotations

from typing import Any

from mlflow.genai.scorers import scorer


@scorer
def tool_usage_recall(expectations: dict, trace: Any) -> float | None:
    """Fraction of expected tools that the agent actually called."""
    expected = expectations.get("expected_tools")
    if not expected:
        return None

    tool_spans = trace.search_spans(span_type="TOOL")
    if not tool_spans:
        return 0.0

    actual = {span.name for span in tool_spans}
    expected_set = set(expected)
    return len(expected_set & actual) / len(expected_set)
