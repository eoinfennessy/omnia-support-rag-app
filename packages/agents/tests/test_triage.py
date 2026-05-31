"""Tests for the triage agent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from omnia_support_agents.models import AgentType, TriageResult, Urgency
from omnia_support_agents.triage import TriageAgent

_PROMPT_TEMPLATE = [{"role": "system", "content": "You are a triage agent."}]
_PROMPT_URI = "prompts:/triage@production"


def _make_completion(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _make_prompt(template: list) -> MagicMock:
    prompt = MagicMock()
    prompt.template = template
    return prompt


def _make_triage_json(
    *,
    urgency: str = "low",
    sub_queries: list[dict] | None = None,
) -> str:
    return json.dumps(
        {
            "urgency": urgency,
            "sub_queries": sub_queries or [{"query": "test query", "agent": "information"}],
        }
    )


def _make_agent(client: MagicMock) -> TriageAgent:
    return TriageAgent(
        completions_client=client,
        model="gpt-4",
        prompt_uri=_PROMPT_URI,
    )


class TestTriageAgent:
    @patch("omnia_support_agents.triage.mlflow.genai.load_prompt")
    def test_simple_information_query(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        client.parse.return_value = _make_completion(
            _make_triage_json(
                urgency="low",
                sub_queries=[{"query": "return policy", "agent": "information"}],
            ),
        )

        result = _make_agent(client).triage("What is your return policy?", [])

        assert isinstance(result, TriageResult)
        assert result.urgency == Urgency.LOW
        assert len(result.sub_queries) == 1
        assert result.sub_queries[0].query == "return policy"
        assert result.sub_queries[0].agent == AgentType.INFORMATION

    @patch("omnia_support_agents.triage.mlflow.genai.load_prompt")
    def test_action_query(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        client.parse.return_value = _make_completion(
            _make_triage_json(
                urgency="medium",
                sub_queries=[{"query": "order status for ORD-2024-001", "agent": "action"}],
            ),
        )

        result = _make_agent(client).triage("What's the status of order ORD-2024-001?", [])

        assert result.urgency == Urgency.MEDIUM
        assert len(result.sub_queries) == 1
        assert result.sub_queries[0].agent == AgentType.ACTION

    @patch("omnia_support_agents.triage.mlflow.genai.load_prompt")
    def test_complex_query_decomposes_into_multiple_sub_queries(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        client.parse.return_value = _make_completion(
            _make_triage_json(
                urgency="high",
                sub_queries=[
                    {"query": "damaged item received", "agent": "action"},
                    {"query": "late delivery", "agent": "action"},
                ],
            ),
        )

        result = _make_agent(client).triage(
            "I ordered 2 items last week, one arrived damaged and the other is late",
            [],
        )

        assert result.urgency == Urgency.HIGH
        assert len(result.sub_queries) == 2
        assert result.sub_queries[0].query == "damaged item received"
        assert result.sub_queries[0].agent == AgentType.ACTION
        assert result.sub_queries[1].query == "late delivery"
        assert result.sub_queries[1].agent == AgentType.ACTION

    @patch("omnia_support_agents.triage.mlflow.genai.load_prompt")
    def test_filters_history_to_user_and_assistant(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        client.parse.return_value = _make_completion(_make_triage_json())
        history = [
            {"role": "developer", "content": "context"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]

        _make_agent(client).triage("follow-up", history)  # ty: ignore[invalid-argument-type]

        messages = client.parse.call_args.kwargs["messages"]
        roles = [m["role"] for m in messages]
        assert "developer" not in roles
        assert roles == ["system", "user", "assistant", "user"]

    @patch("omnia_support_agents.triage.mlflow.genai.load_prompt")
    def test_passes_response_format(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        client.parse.return_value = _make_completion(_make_triage_json())

        _make_agent(client).triage("question", [])

        assert client.parse.call_args.kwargs["response_format"] is TriageResult

    @patch("omnia_support_agents.triage.mlflow.genai.load_prompt")
    def test_uses_temperature_zero(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        client.parse.return_value = _make_completion(_make_triage_json())

        _make_agent(client).triage("question", [])

        assert client.parse.call_args.kwargs["temperature"] == 0

    @patch("omnia_support_agents.triage.mlflow.genai.load_prompt")
    def test_raises_on_empty_content(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = None
        client.parse.return_value = response

        with pytest.raises(ValueError, match="empty content"):
            _make_agent(client).triage("question", [])


class TestTriageResult:
    def test_rejects_empty_sub_queries(self):
        with pytest.raises(ValidationError):
            TriageResult(urgency=Urgency.LOW, sub_queries=[])
