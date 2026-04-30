"""Tests for query reformulation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from omnia_support_rag.reformulate import reformulate_query


def _make_completion(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _make_prompt(template: list) -> MagicMock:
    prompt = MagicMock()
    prompt.template = template
    return prompt


_PROMPT_TEMPLATE = [{"role": "system", "content": "Rewrite the question."}]
_PROMPT_URI = "prompts:/reformulate_query@production"


class TestReformulateQuery:
    def test_returns_question_unchanged_when_no_history(self):
        client = MagicMock()

        result = reformulate_query(
            [],
            "What is your return policy?",
            completions_client=client,
            model="gpt-4",
            prompt_uri=_PROMPT_URI,
        )

        assert result == "What is your return policy?"
        client.create.assert_not_called()

    @patch("omnia_support_rag.reformulate.mlflow.genai.load_prompt")
    def test_calls_llm_with_history(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        client.create.return_value = _make_completion("What is the warranty policy for the Dell XPS 15?")
        history = [
            {"role": "developer", "content": "context about Dell XPS"},
            {"role": "user", "content": "Tell me about the Dell XPS 15"},
            {"role": "assistant", "content": "The Dell XPS 15 is a laptop..."},
        ]

        result = reformulate_query(
            history,  # ty: ignore[invalid-argument-type]
            "What about the warranty?",
            completions_client=client,
            model="gpt-4",
            prompt_uri=_PROMPT_URI,
        )

        assert result == "What is the warranty policy for the Dell XPS 15?"
        mock_load_prompt.assert_called_once_with(_PROMPT_URI)

    @patch("omnia_support_rag.reformulate.mlflow.genai.load_prompt")
    def test_filters_to_user_and_assistant_roles(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        client.create.return_value = _make_completion("standalone question")
        history = [
            {"role": "developer", "content": "context"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]

        reformulate_query(
            history,  # ty: ignore[invalid-argument-type]
            "follow-up",
            completions_client=client,
            model="gpt-4",
            prompt_uri=_PROMPT_URI,
        )

        messages = client.create.call_args.kwargs["messages"]
        roles = [m["role"] for m in messages]
        assert "developer" not in roles
        assert roles == ["system", "user", "assistant", "user"]

    @patch("omnia_support_rag.reformulate.mlflow.genai.load_prompt")
    def test_returns_original_question_when_llm_returns_none(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = None
        client.create.return_value = response

        result = reformulate_query(
            [{"role": "user", "content": "prior"}],
            "follow-up",
            completions_client=client,
            model="gpt-4",
            prompt_uri=_PROMPT_URI,
        )

        assert result == "follow-up"
