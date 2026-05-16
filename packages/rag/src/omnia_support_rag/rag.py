"""RAG retrieval pipeline.

Provides the end-to-end retrieve -> build prompt -> generate flow,
instrumented with MLflow tracing so that retrieval scorers can
inspect RETRIEVER spans automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import mlflow
from jinja2 import Template

from omnia_support_models import ProductCategory
from omnia_support_store import ScoredChunk  # noqa: TC001 — used in dataclass field at runtime

from .query_interpretation import PassthroughQueryInterpreter, QueryInterpreter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mlflow.entities.model_registry.prompt_version import PromptVersion
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

    from omnia_support_store import WhereFilter

    from .context_processing import ContextProcessor
    from .query_interpretation import InterpretedQuery


FALLBACK_ANSWER = (
    "I'm sorry, I wasn't able to find any relevant information to answer your question. "
    "Could you try rephrasing or providing more details?"
)


@dataclass(frozen=True)
class RagResult:
    """Result from the RAG pipeline."""

    answer: str
    retrieved_chunks: list[ScoredChunk]


class Rescorer(Protocol):
    """Protocol for a post-retrieval rescoring strategy."""

    def rescore(self, chunk: ScoredChunk, intent: InterpretedQuery | None = None) -> ScoredChunk:
        """Return a new ScoredChunk with an adjusted score."""
        ...


class CompletionsClient(Protocol):
    """Protocol for an OpenAI-compatible chat completions client."""

    def create(
        self,
        *,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float,
        response_format: Any = None,
    ) -> ChatCompletion:
        """Create a chat completion."""
        ...

    def parse(
        self,
        *,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float,
        response_format: Any = None,
    ) -> ChatCompletion:
        """Create a chat completion with structured output parsing."""
        ...


class Retriever(Protocol):
    """Protocol for a retrieval strategy."""

    def retrieve(self, query: str, n_results: int, where: WhereFilter | None = None) -> list[ScoredChunk]:
        """Retrieve relevant documents for the given query."""
        ...


def _apply_rescorers(
    chunks: list[ScoredChunk],
    rescorers: Sequence[Rescorer],
    intent: InterpretedQuery | None = None,
) -> list[ScoredChunk]:
    """Apply rescorers in sequence."""
    for rescorer in rescorers:
        chunks = [rescorer.rescore(r, intent) for r in chunks]
    return chunks


@mlflow.trace(name="reorder_by_score")
def _reorder_by_score(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    """Sort chunks by score descending."""
    return sorted(chunks, key=lambda r: r.score, reverse=True)


def render_messages(
    prompt: PromptVersion,
    *,
    history: list[ChatCompletionMessageParam] | None = None,
    records: Sequence[ScoredChunk] = (),
    **variables: str,
) -> list[ChatCompletionMessageParam]:
    """Render a prompt template and splice in conversation history.

    Produces: [system, *history, developer, user] when history is provided,
    or the standard [system, developer, user] when it is not.
    """
    all_variables = {**variables, "context": join_records(records)}
    rendered: list[ChatCompletionMessageParam] = [  # ty: ignore[invalid-assignment]
        {**msg, "content": Template(msg["content"]).render(**all_variables)}
        for msg in prompt.template
        if isinstance(msg, dict)
    ]
    prior_turns = [m for m in (history or []) if m.get("role") != "system"]
    return [*rendered[:1], *prior_turns, *rendered[1:]]


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


def join_records(records: Sequence[ScoredChunk]) -> str:
    """Join chunk contents into a single context string."""
    return "\n---\n".join(r.page_content for r in records)


def _build_where_filter(product_categories: Sequence[str]) -> WhereFilter | None:
    """Build a where filter from product categories."""
    if not product_categories:
        return None
    categories = {*product_categories, ProductCategory.ALL, ProductCategory.NONE}
    return {"product_category": {"$in": list(categories)}}


@mlflow.trace(name="rag_pipeline")
def run_rag_pipeline(
    question: str,
    *,
    completions_client: CompletionsClient,
    retriever: Retriever,
    prompt_uri: str,
    n_results: int,
    model: str,
    history: list[ChatCompletionMessageParam] | None = None,
    query_interpreter: QueryInterpreter = PassthroughQueryInterpreter(),  # noqa: B008 — stateless
    rescorers: Sequence[Rescorer] | None = None,
    context_processors: Sequence[ContextProcessor] | None = None,
    use_metadata_filter: bool = True,
) -> RagResult:
    """Run the full RAG pipeline.

    1. Interpret query
    2. Retrieve
    3. Rescore
    4. Reorder by score
    5. Process context
    6. Build prompt
    7. Generate
    """
    intent = query_interpreter.interpret(question, history or [])

    where = _build_where_filter(intent.product_categories) if use_metadata_filter else None
    chunks = retriever.retrieve(intent.effective_retrieval_query, n_results, where=where)

    if not chunks:
        return RagResult(answer=FALLBACK_ANSWER, retrieved_chunks=[])

    if rescorers:
        chunks = _apply_rescorers(chunks, rescorers, intent)

    chunks = _reorder_by_score(chunks)

    for processor in context_processors or ():
        chunks = processor.process(chunks)

    prompt = mlflow.genai.load_prompt(prompt_uri)
    messages = render_messages(prompt, history=history, records=chunks, question=question)

    answer = generate_answer(completions_client, messages, model)

    return RagResult(answer=answer, retrieved_chunks=chunks)
