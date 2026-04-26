"""RAG retrieval pipeline.

Provides the end-to-end retrieve -> build prompt -> generate flow,
instrumented with MLflow tracing so that retrieval scorers can
inspect RETRIEVER spans automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import mlflow
from jinja2 import Template

if TYPE_CHECKING:
    from mlflow.entities.model_registry.prompt_version import PromptVersion
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

    from omnia_support_vectorstore import VectorStore


class CompletionsClient(Protocol):
    """Protocol for an OpenAI-compatible chat completions client."""

    def create(
        self,
        *,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float,
    ) -> ChatCompletion:
        """Create a chat completion."""
        ...


@mlflow.trace(span_type="RETRIEVER")
def retrieve_context(
    store: VectorStore,
    query: str,
    n_results: int,
) -> list[dict]:
    """Query the vector store and return results as dicts for MLflow scorers."""
    records = store.query(query, n_results)
    return [{"page_content": r.chunk, "metadata": r.metadata} for r in records]


def render_messages(
    prompt: PromptVersion,
    **variables: str,
) -> list[ChatCompletionMessageParam]:
    """Render a prompt template by substituting {{variables}} in message content."""
    return [  # ty: ignore[invalid-return-type]
        {**msg, "content": Template(msg["content"]).render(**variables)}
        for msg in prompt.template
        if isinstance(msg, dict)
    ]


@mlflow.trace(span_type="LLM")
def generate_answer(
    completions_client: CompletionsClient,
    messages: list[ChatCompletionMessageParam],
    model: str,
) -> str:
    """Call the completions client and return the response text."""
    response = completions_client.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content or ""


@mlflow.trace(name="rag_pipeline")
def run_rag_pipeline(
    question: str,
    *,
    completions_client: CompletionsClient,
    store: VectorStore,
    prompt_uri: str,
    n_results: int,
    model: str,
) -> dict:
    """Run the full RAG pipeline: retrieve -> build prompt -> generate."""
    prompt = mlflow.genai.load_prompt(prompt_uri)
    records = retrieve_context(store, question, n_results)
    chunks = [r["page_content"] for r in records]
    context = "\n---\n".join(chunks)
    messages = render_messages(prompt, context=context, question=question)
    answer = generate_answer(completions_client, messages, model)

    return {
        "answer": answer,
        "retrieved_context": context,
        "retrieved_records": records,
    }
