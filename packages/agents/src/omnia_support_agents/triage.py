"""Triage agent: classifies urgency and decomposes queries via structured LLM output."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow

from .models import TriageDecision

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

    from omnia_support_rag.rag import CompletionsClient


class TriageAgent:
    """Classifies query urgency and decomposes into routed sub-queries."""

    def __init__(
        self,
        *,
        completions_client: CompletionsClient,
        model: str,
        prompt_uri: str,
    ) -> None:
        """Create a triage agent."""
        self._client = completions_client
        self._model = model
        self._prompt_uri = prompt_uri

    @mlflow.trace(name="triage")
    def triage(
        self,
        question: str,
        history: list[ChatCompletionMessageParam],
    ) -> TriageDecision:
        """Classify urgency and decompose the question into sub-queries."""
        prompt = mlflow.genai.load_prompt(self._prompt_uri)
        system_messages: list[ChatCompletionMessageParam] = [
            {"role": msg["role"], "content": msg["content"]} for msg in prompt.template if isinstance(msg, dict)
        ]
        user_assistant_turns: list[ChatCompletionMessageParam] = [
            m for m in history if m.get("role") in ("user", "assistant")
        ]
        messages: list[ChatCompletionMessageParam] = [
            *system_messages,
            *user_assistant_turns,
            {"role": "user", "content": question},
        ]
        response = self._client.parse(
            model=self._model,
            messages=messages,
            temperature=0,
            response_format=TriageDecision,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned empty content for triage")
        return TriageDecision.model_validate_json(content)
