"""Tests for query interpretation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from omnia_support_rag.query_interpretation import InterpretedQuery, LLMQueryInterpreter


def _make_completion(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _make_prompt(template: list) -> MagicMock:
    prompt = MagicMock()
    prompt.template = template
    return prompt


_PROMPT_TEMPLATE = [{"role": "system", "content": "Interpret the query."}]
_PROMPT_URI = "prompts:/interpret_query@production"


def _make_intent_json(
    *,
    query: str = "standalone question",
    product_categories: list[str] | None = None,
    policy: int = 5,
    faq: int = 5,
    support_ticket: int = 5,
) -> str:
    return json.dumps(
        {
            "query": query,
            "product_categories": product_categories or [],
            "doc_type_relevance": {
                "policy": policy,
                "faq": faq,
                "support_ticket": support_ticket,
            },
        }
    )


class TestInterpretedQuery:
    def test_effective_retrieval_query_falls_back_to_query(self):
        iq = InterpretedQuery(query="clean question")
        assert iq.effective_retrieval_query == "clean question"

    def test_effective_retrieval_query_uses_retrieval_query_when_set(self):
        iq = InterpretedQuery(query="clean question", retrieval_query="clean question synonyms related terms")
        assert iq.effective_retrieval_query == "clean question synonyms related terms"

    def test_effective_retrieval_query_falls_back_when_empty_string(self):
        iq = InterpretedQuery(query="clean question", retrieval_query="")
        assert iq.effective_retrieval_query == "clean question"


class TestLLMQueryInterpreter:
    @patch("omnia_support_rag.query_interpretation.mlflow.genai.load_prompt")
    def test_returns_interpreted_query(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        client.parse.return_value = _make_completion(
            _make_intent_json(
                query="What is the return policy for laptops?",
                product_categories=["laptops"],
                policy=9,
                faq=3,
                support_ticket=2,
            ),
        )
        interpreter = LLMQueryInterpreter(
            completions_client=client,
            model="gpt-4",
            prompt_uri=_PROMPT_URI,
        )

        result = interpreter.interpret("return policy?", [])

        assert isinstance(result, InterpretedQuery)
        assert result.query == "What is the return policy for laptops?"
        assert result.product_categories == ["laptops"]
        assert result.doc_type_relevance.policy == 9
        assert result.doc_type_relevance.faq == 3
        assert result.doc_type_relevance.support_ticket == 2

    @patch("omnia_support_rag.query_interpretation.mlflow.genai.load_prompt")
    def test_always_calls_llm_even_without_history(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        client.parse.return_value = _make_completion(_make_intent_json())
        interpreter = LLMQueryInterpreter(
            completions_client=client,
            model="gpt-4",
            prompt_uri=_PROMPT_URI,
        )

        interpreter.interpret("first question", [])

        client.parse.assert_called_once()

    @patch("omnia_support_rag.query_interpretation.mlflow.genai.load_prompt")
    def test_filters_history_to_user_and_assistant(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        client.parse.return_value = _make_completion(_make_intent_json())
        interpreter = LLMQueryInterpreter(
            completions_client=client,
            model="gpt-4",
            prompt_uri=_PROMPT_URI,
        )
        history = [
            {"role": "developer", "content": "context"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]

        interpreter.interpret("follow-up", history)  # ty: ignore[invalid-argument-type]

        messages = client.parse.call_args.kwargs["messages"]
        roles = [m["role"] for m in messages]
        assert "developer" not in roles
        assert roles == ["system", "user", "assistant", "user"]

    @patch("omnia_support_rag.query_interpretation.mlflow.genai.load_prompt")
    def test_empty_product_categories(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        client.parse.return_value = _make_completion(
            _make_intent_json(query="general question", product_categories=[]),
        )
        interpreter = LLMQueryInterpreter(
            completions_client=client,
            model="gpt-4",
            prompt_uri=_PROMPT_URI,
        )

        result = interpreter.interpret("general question", [])

        assert result.product_categories == []

    @patch("omnia_support_rag.query_interpretation.mlflow.genai.load_prompt")
    def test_passes_response_format(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(_PROMPT_TEMPLATE)
        client = MagicMock()
        client.parse.return_value = _make_completion(_make_intent_json())
        interpreter = LLMQueryInterpreter(
            completions_client=client,
            model="gpt-4",
            prompt_uri=_PROMPT_URI,
        )

        interpreter.interpret("question", [])

        assert client.parse.call_args.kwargs["response_format"] is InterpretedQuery
