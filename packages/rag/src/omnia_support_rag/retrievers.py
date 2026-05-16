"""Retriever implementations: semantic, BM25, and hybrid (RRF) search."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol

import mlflow

from omnia_support_store import ScoredChunk

from .rag import Retriever  # noqa: TC001 — used in NamedTuple field at runtime

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omnia_support_store import KeywordStore, VectorStore, WhereFilter


class RetrievalFilter(Protocol):
    """Protocol for a post-retrieval filter on raw store chunks."""

    def filter(self, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        """Return a filtered subset of chunks."""
        ...


def _apply_filters(
    chunks: list[ScoredChunk],
    filters: Sequence[RetrievalFilter],
) -> list[ScoredChunk]:
    """Apply filters in sequence, each narrowing the result set."""
    for f in filters:
        chunks = f.filter(chunks)
    return chunks


def _rrf_fuse(
    *weighted_rankings: tuple[float, list[ScoredChunk]],
    k: int = 60,
    n_results: int = 5,
) -> list[ScoredChunk]:
    """Fuse multiple weighted rankings using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    records: dict[str, ScoredChunk] = {}
    for weight, ranking in weighted_rankings:
        for rank, record in enumerate(ranking):
            scores[record.id] = scores.get(record.id, 0.0) + weight / (k + rank + 1)
            records.setdefault(record.id, record)
    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:n_results]
    return [
        ScoredChunk(id=rid, page_content=records[rid].page_content, metadata=records[rid].metadata, score=scores[rid])
        for rid in sorted_ids
    ]


class SemanticRetriever:
    """Retriever that wraps a VectorStore for pure semantic search."""

    def __init__(
        self,
        store: VectorStore,
        *,
        filters: Sequence[RetrievalFilter] | None = None,
        overfetch_factor: int = 3,
    ) -> None:
        """Wrap a vector store for pure semantic retrieval."""
        self._store = store
        self._filters = filters
        self._overfetch_factor = overfetch_factor

    @mlflow.trace(span_type="RETRIEVER")
    def retrieve(self, query: str, n_results: int, where: WhereFilter | None = None) -> list[ScoredChunk]:
        """Retrieve documents using semantic similarity via the vector store."""
        fetch_n = n_results * self._overfetch_factor if self._filters else n_results
        records = self._store.query(query, fetch_n, where=where)
        if self._filters:
            records = _apply_filters(records, self._filters)
        return records[:n_results]


class BM25Retriever:
    """Retriever using BM25 keyword search via a KeywordStore."""

    def __init__(
        self,
        keyword_store: KeywordStore,
        *,
        filters: Sequence[RetrievalFilter] | None = None,
        overfetch_factor: int = 3,
    ) -> None:
        """Wrap a keyword store for BM25 retrieval."""
        self._keyword_store = keyword_store
        self._filters = filters
        self._overfetch_factor = overfetch_factor

    @mlflow.trace(span_type="RETRIEVER")
    def retrieve(self, query: str, n_results: int, where: WhereFilter | None = None) -> list[ScoredChunk]:
        """Retrieve documents using BM25 keyword matching."""
        fetch_n = n_results * self._overfetch_factor if self._filters else n_results
        records = self._keyword_store.query(query, fetch_n, where=where)
        if self._filters:
            records = _apply_filters(records, self._filters)
        return records[:n_results]


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
    def retrieve(self, query: str, n_results: int, where: WhereFilter | None = None) -> list[ScoredChunk]:
        """Retrieve documents using RRF fusion of multiple retriever sources."""
        rankings = [
            (source.weight, source.retriever.retrieve(query, n_results, where=where)) for source in self._sources
        ]
        return _rrf_fuse(*rankings, k=self._rrf_k, n_results=n_results)
