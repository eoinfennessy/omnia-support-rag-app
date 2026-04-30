"""Reformulate follow-up questions into standalone queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

    from .rag import CompletionsClient


@mlflow.trace(name="reformulate_query")
def reformulate_query(
    history: list[ChatCompletionMessageParam],
    question: str,
    *,
    completions_client: CompletionsClient,
    model: str,
    prompt_uri: str,
) -> str:
    """Rewrite a follow-up question as a standalone query.

    Skips the LLM call entirely when history is empty (first turn).
    """
    if not history:
        return question

    prompt = mlflow.genai.load_prompt(prompt_uri)
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
    response = completions_client.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content or question
