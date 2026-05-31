"""Aggregator agent: synthesises a single answer from multiple sub-query results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openai.types.chat import ChatCompletionMessageParam

    from omnia_support_rag.rag import CompletionsClient

    from .models import SubQueryResult

_RESULT_SEPARATOR = "\n\n---\n\n"


class AggregatorAgent:
    """Synthesises a single coherent answer from multiple sub-query results."""

    def __init__(
        self,
        *,
        completions_client: CompletionsClient,
        model: str,
        prompt_uri: str,
    ) -> None:
        """Create an aggregator agent."""
        self._client = completions_client
        self._model = model
        self._prompt_uri = prompt_uri

    @mlflow.trace(name="aggregate")
    def aggregate(self, results: Sequence[SubQueryResult]) -> str:
        """Aggregate sub-query results into a single answer."""
        if not results:
            raise ValueError("Aggregator received no results")

        prompt = mlflow.genai.load_prompt(self._prompt_uri)
        system_messages: list[ChatCompletionMessageParam] = [
            {"role": msg["role"], "content": msg["content"]} for msg in prompt.template if isinstance(msg, dict)
        ]

        formatted = _RESULT_SEPARATOR.join(r.format_for_aggregator() for r in results)

        messages: list[ChatCompletionMessageParam] = [
            *system_messages,
            {"role": "developer", "content": formatted},
        ]

        response = self._client.create(
            model=self._model,
            messages=messages,
            temperature=0,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned empty content for aggregation")
        return content
