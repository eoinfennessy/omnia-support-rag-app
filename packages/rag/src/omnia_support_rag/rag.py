"""RAG retrieval pipeline.

Provides the end-to-end retrieve -> build prompt -> generate flow,
instrumented with MLflow tracing so that retrieval scorers can
inspect RETRIEVER spans automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol

import mlflow
from jinja2 import Template

from .reformulate import reformulate_query

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mlflow.entities.model_registry.prompt_version import PromptVersion
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

    from omnia_support_store import KeywordStore, ScoredChunk, VectorStore

    from .context_formatting import ContextFormatter
    from .filters import RetrievalFilter


class Rescorer(Protocol):
    """Protocol for a post-retrieval rescoring strategy."""

    def rescore(self, chunk: ScoredChunk) -> ScoredChunk:
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
    ) -> ChatCompletion:
        """Create a chat completion."""
        ...


class Retriever(Protocol):
    """Protocol for a retrieval strategy."""

    def retrieve(self, query: str, n_results: int) -> list[ScoredChunk]:
        """Retrieve relevant documents for the given query."""
        ...


def _apply_rescorers(
    records: list[ScoredChunk],
    rescorers: Sequence[Rescorer],
) -> list[ScoredChunk]:
    """Apply rescorers in sequence and re-sort by score (descending)."""
    for rescorer in rescorers:
        records = [rescorer.rescore(r) for r in records]
    records.sort(key=lambda r: r.score, reverse=True)
    return records


def _apply_filters(
    records: list[ScoredChunk],
    filters: Sequence[RetrievalFilter],
) -> list[ScoredChunk]:
    """Apply filters in sequence, each narrowing the result set."""
    for f in filters:
        records = f.filter(records)
    return records


class SemanticRetriever:
    """Retriever that wraps a VectorStore for pure semantic search."""

    def __init__(
        self,
        store: VectorStore,
        *,
        filters: Sequence[RetrievalFilter] | None = None,
    ) -> None:
        """Wrap a vector store for pure semantic retrieval."""
        self._store = store
        self._filters = filters

    @mlflow.trace(span_type="RETRIEVER")
    def retrieve(self, query: str, n_results: int) -> list[ScoredChunk]:
        """Retrieve documents using semantic similarity via the vector store."""
        records = self._store.query(query, n_results)
        if self._filters:
            records = _apply_filters(records, self._filters)
        return records


class BM25Retriever:
    """Retriever using BM25 keyword search via a KeywordStore."""

    def __init__(
        self,
        keyword_store: KeywordStore,
        *,
        filters: Sequence[RetrievalFilter] | None = None,
    ) -> None:
        """Wrap a keyword store for BM25 retrieval."""
        self._keyword_store = keyword_store
        self._filters = filters

    @mlflow.trace(span_type="RETRIEVER")
    def retrieve(self, query: str, n_results: int) -> list[ScoredChunk]:
        """Retrieve documents using BM25 keyword matching."""
        records = self._keyword_store.query(query, n_results)
        if self._filters:
            records = _apply_filters(records, self._filters)
        return records


class RetrieverWeight(NamedTuple):
    """A retriever paired with its RRF fusion weight."""

    retriever: Retriever
    weight: float = 1.0


class HybridRetriever:
    """Retriever combining multiple sources via Reciprocal Rank Fusion."""

    def __init__(
        self,
        *sources: RetrieverWeight,
        rrf_k: int = 60,
    ) -> None:
        """Fuse multiple weighted retriever sources via RRF."""
        self._sources = sources
        self._rrf_k = rrf_k

    @mlflow.trace(span_type="RETRIEVER")
    def retrieve(self, query: str, n_results: int) -> list[ScoredChunk]:
        """Retrieve documents using RRF fusion of multiple retriever sources."""
        rankings = [(source.weight, source.retriever.retrieve(query, n_results)) for source in self._sources]
        return _rrf_fuse(*rankings, k=self._rrf_k, n_results=n_results)


def _rrf_fuse(
    *weighted_rankings: tuple[float, list[ScoredChunk]],
    k: int = 60,
    n_results: int = 5,
) -> list[ScoredChunk]:
    """Fuse multiple weighted rankings using Reciprocal Rank Fusion."""
    from omnia_support_store import ScoredChunk as _ScoredChunk

    scores: dict[str, float] = {}
    records: dict[str, ScoredChunk] = {}
    for weight, ranking in weighted_rankings:
        for rank, record in enumerate(ranking):
            scores[record.id] = scores.get(record.id, 0.0) + weight / (k + rank + 1)
            records.setdefault(record.id, record)
    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:n_results]
    return [
        _ScoredChunk(id=rid, chunk=records[rid].chunk, metadata=records[rid].metadata, score=scores[rid])
        for rid in sorted_ids
    ]


@mlflow.trace(span_type="RETRIEVER")
def retrieve_context(
    store: VectorStore,
    query: str,
    n_results: int,
) -> list[dict]:
    """Query the vector store and return results as dicts for MLflow scorers."""
    records = store.query(query, n_results)
    return [{"page_content": r.chunk, "metadata": r.metadata, "score": r.score} for r in records]


def render_messages(
    prompt: PromptVersion,
    *,
    history: list[ChatCompletionMessageParam] | None = None,
    **variables: str,
) -> list[ChatCompletionMessageParam]:
    """Render a prompt template and splice in conversation history.

    Produces: [system, *history, developer, user] when history is provided,
    or the standard [system, developer, user] when it is not.
    """
    rendered: list[ChatCompletionMessageParam] = [  # ty: ignore[invalid-assignment]
        {**msg, "content": Template(msg["content"]).render(**variables)}
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


def _to_dicts(records: list[ScoredChunk]) -> list[dict]:
    """Convert ScoredChunks to the dict format used downstream."""
    return [{"page_content": r.chunk, "metadata": r.metadata, "score": r.score} for r in records]


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
    reformulate_prompt_uri: str | None = None,
    rescorers: Sequence[Rescorer] | None = None,
    context_formatters: Sequence[ContextFormatter] | None = None,
) -> dict:
    """Run the full RAG pipeline: reformulate -> retrieve -> rescore -> format -> build prompt -> generate."""
    if reformulate_prompt_uri:
        question = reformulate_query(
            history or [],
            question,
            completions_client=completions_client,
            model=model,
            prompt_uri=reformulate_prompt_uri,
        )

    prompt = mlflow.genai.load_prompt(prompt_uri)
    scored_chunks = retriever.retrieve(question, n_results)
    if rescorers:
        scored_chunks = _apply_rescorers(scored_chunks, rescorers)
    records = _to_dicts(scored_chunks)
    for formatter in context_formatters or ():
        records = formatter.format(records)
    chunks = [r["page_content"] for r in records]
    context = "\n---\n".join(chunks)
    rendered = render_messages(prompt, history=history, context=context, question=question)
    answer = generate_answer(completions_client, rendered, model)

    return {
        "answer": answer,
        "retrieved_context": context,
        "retrieved_records": records,
    }
