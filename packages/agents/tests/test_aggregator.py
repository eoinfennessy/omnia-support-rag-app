"""Tests for the aggregator agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from omnia_support_agents.aggregator import AggregatorAgent
from omnia_support_agents.models import ActionResult, InformationResult

_PROMPT_TEMPLATE = [{"role": "system", "content": "You are a response aggregator."}]
_PROMPT_URI = "prompts:/aggregate@production"


def _make_completion(content: str | None = "Combined answer") -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _make_prompt(template: list) -> MagicMock:
    prompt = MagicMock()
    prompt.template = template
    return prompt


def _make_agent(client: MagicMock) -> AggregatorAgent:
    return AggregatorAgent(
        completions_client=client,
        model="gpt-4",
        prompt_uri=_PROMPT_URI,
    )


def _make_information_result(
    query: str = "test query",
    answer: str = "test answer",
) -> InformationResult:
    return InformationResult(query=query, answer=answer)


def _make_action_result(
    query: str = "test query",
    answer: str = "test answer",
) -> ActionResult:
    return ActionResult(query=query, answer=answer)


class TestAggregatorAgent:
    @patch("omnia_support_agents.aggregator.mlflow.genai.load_prompt")
    def test_single_result_calls_llm(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        client.create.return_value = _make_completion("aggregated answer")

        answer = _make_agent(client).aggregate(
            [
                _make_information_result(answer="raw answer"),
            ]
        )

        assert answer == "aggregated answer"
        client.create.assert_called_once()

    @patch("omnia_support_agents.aggregator.mlflow.genai.load_prompt")
    def test_multiple_results_calls_llm(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        client.create.return_value = _make_completion("synthesised answer")

        answer = _make_agent(client).aggregate(
            [
                _make_information_result(answer="first"),
                _make_action_result(answer="second"),
            ]
        )

        assert answer == "synthesised answer"
        client.create.assert_called_once()

    @patch("omnia_support_agents.aggregator.mlflow.genai.load_prompt")
    def test_multiple_results_formats_each_result(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        client.create.return_value = _make_completion()

        _make_agent(client).aggregate(
            [
                _make_information_result(query="q1", answer="a1"),
                _make_action_result(query="q2", answer="a2"),
            ]
        )

        developer_message = client.create.call_args.kwargs["messages"][-1]
        assert developer_message["role"] == "developer"
        assert "q1" in developer_message["content"]
        assert "a1" in developer_message["content"]
        assert "q2" in developer_message["content"]
        assert "a2" in developer_message["content"]

    @patch("omnia_support_agents.aggregator.mlflow.genai.load_prompt")
    def test_includes_system_messages_from_prompt(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        client.create.return_value = _make_completion()

        _make_agent(client).aggregate(
            [
                _make_information_result(),
                _make_action_result(),
            ]
        )

        messages = client.create.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a response aggregator."

    @patch("omnia_support_agents.aggregator.mlflow.genai.load_prompt")
    def test_uses_temperature_zero(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        client.create.return_value = _make_completion()

        _make_agent(client).aggregate(
            [
                _make_information_result(),
                _make_action_result(),
            ]
        )

        assert client.create.call_args.kwargs["temperature"] == 0

    @patch("omnia_support_agents.aggregator.mlflow.genai.load_prompt")
    def test_uses_create_not_parse(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        client.create.return_value = _make_completion()

        _make_agent(client).aggregate(
            [
                _make_information_result(),
                _make_action_result(),
            ]
        )

        client.create.assert_called_once()
        client.parse.assert_not_called()

    def test_empty_results_raises(self):
        client = MagicMock()

        with pytest.raises(ValueError, match="no results"):
            _make_agent(client).aggregate([])

    @patch("omnia_support_agents.aggregator.mlflow.genai.load_prompt")
    def test_raises_on_empty_content(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        client.create.return_value = _make_completion(None)

        with pytest.raises(ValueError, match="empty content"):
            _make_agent(client).aggregate(
                [
                    _make_information_result(),
                    _make_action_result(),
                ]
            )

    @patch("omnia_support_agents.aggregator.mlflow.genai.load_prompt")
    def test_mixed_result_types(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        client.create.return_value = _make_completion("mixed answer")

        answer = _make_agent(client).aggregate(
            [
                _make_information_result(query="info q", answer="info a"),
                _make_action_result(query="action q", answer="action a"),
            ]
        )

        assert answer == "mixed answer"
        developer_message = client.create.call_args.kwargs["messages"][-1]["content"]
        assert "info q" in developer_message
        assert "action q" in developer_message
