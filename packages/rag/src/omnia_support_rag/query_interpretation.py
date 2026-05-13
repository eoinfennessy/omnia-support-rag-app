"""Query interpretation: rewrite + classify in a single LLM call."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

import mlflow
from pydantic import BaseModel, Field

from omnia_support_models import ProductCategory

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

    from .rag import CompletionsClient

FilterableCategory = Literal[
    ProductCategory.LAPTOPS,
    ProductCategory.MONITORS,
    ProductCategory.ACCESSORIES,
    ProductCategory.STORAGE,
]


class DocumentTypeRelevance(BaseModel):
    """Per-document-type relevance weights returned by query interpretation."""

    policy: int = Field(ge=0, le=10)
    faq: int = Field(ge=0, le=10)
    support_ticket: int = Field(ge=0, le=10)


class InterpretedQuery(BaseModel):
    """Structured result of query interpretation."""

    query: str
    product_categories: list[FilterableCategory] = Field(default_factory=list)
    doc_type_relevance: DocumentTypeRelevance


class QueryInterpreter(Protocol):
    """Protocol for interpreting a user query into structured intent."""

    def interpret(
        self,
        question: str,
        history: list[ChatCompletionMessageParam],
    ) -> InterpretedQuery:
        """Interpret a user question, returning a rewritten query with metadata."""
        ...


class LLMQueryInterpreter:
    """QueryInterpreter that uses an LLM with structured output."""

    def __init__(
        self,
        *,
        completions_client: CompletionsClient,
        model: str,
        prompt_uri: str,
    ) -> None:
        """Create an LLM-backed query interpreter."""
        self._client = completions_client
        self._model = model
        self._prompt_uri = prompt_uri

    @mlflow.trace(name="interpret_query")
    def interpret(
        self,
        question: str,
        history: list[ChatCompletionMessageParam],
    ) -> InterpretedQuery:
        """Interpret a query using an LLM with structured output."""
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
        response = self._client.create(
            model=self._model,
            messages=messages,
            temperature=0,
            response_format=InterpretedQuery,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned empty content for query interpretation")
        return InterpretedQuery.model_validate_json(content)
