"""BM25 keyword store backed by the bm25s library."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import bm25s

from ..keyword_store import KeywordStore
from ..models import ScoredChunk

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from ..models import Chunk, WhereFilter

_RECORDS_FILENAME = "records.json"


def _matches_where(metadata: Mapping[str, Any], where: WhereFilter) -> bool:
    """Check if metadata matches all where-clause conditions.

    Supports exact equality and ``$in`` membership:
    - ``{"key": "value"}`` — exact match
    - ``{"key": {"$in": ["a", "b"]}}`` — membership check
    """
    for key, condition in where.items():
        value = metadata.get(key)
        if isinstance(condition, dict) and "$in" in condition:
            if value not in condition["$in"]:
                return False
        elif value != condition:
            return False
    return True


class BM25sKeywordStore(KeywordStore):
    """Keyword store using BM25 scoring with disk persistence."""

    def __init__(self, *, index_path: Path | None = None) -> None:
        """Create a BM25 keyword store, loading from *index_path* if it exists."""
        self._index_path = index_path
        self._retriever: bm25s.BM25 | None = None
        self._records: list[Chunk] = []
        if index_path and (index_path / "params.index.json").exists():
            self._load()

    def index(self, records: list[Chunk]) -> None:  # noqa: D102
        self._records = list(records)
        corpus = [r.chunk for r in records]
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
        self._retriever = bm25s.BM25()
        self._retriever.index(corpus_tokens)
        if self._index_path:
            self._save()

    def query(  # noqa: D102
        self,
        query_text: str,
        n_results: int,
        where: WhereFilter | None = None,
    ) -> list[ScoredChunk]:
        if not self._retriever or not self._records:
            return []
        k = min(self.count(), n_results * 3 if where else n_results)
        query_tokens = bm25s.tokenize([query_text], stopwords="en")
        indices, scores = self._retriever.retrieve(query_tokens, k=k)
        scored = [
            ScoredChunk(
                id=self._records[idx].id,
                chunk=self._records[idx].chunk,
                metadata=self._records[idx].metadata,
                score=float(score),
            )
            for idx, score in zip(indices[0], scores[0], strict=True)
        ]
        if where:
            scored = [r for r in scored if _matches_where(r.metadata, where)]
        return scored[:n_results]

    def reset(self) -> None:  # noqa: D102
        self._retriever = None
        self._records = []

    def count(self) -> int:  # noqa: D102
        return len(self._records)

    def _save(self) -> None:
        if self._index_path is None or self._retriever is None:
            raise RuntimeError("Cannot save: index_path and retriever must be set")
        self._index_path.mkdir(parents=True, exist_ok=True)
        self._retriever.save(str(self._index_path))
        records_path = self._index_path / _RECORDS_FILENAME
        records_path.write_text(
            json.dumps(
                [{"id": r.id, "chunk": r.chunk, "metadata": dict(r.metadata)} for r in self._records],
            ),
            encoding="utf-8",
        )

    def _load(self) -> None:
        if self._index_path is None:
            raise RuntimeError("Cannot load: index_path must be set")
        from ..models import Chunk, ChunkMetadata

        self._retriever = bm25s.BM25.load(str(self._index_path))
        records_path = self._index_path / _RECORDS_FILENAME
        raw = json.loads(records_path.read_text(encoding="utf-8"))
        self._records = [
            Chunk(
                id=entry["id"],
                chunk=entry["chunk"],
                metadata=ChunkMetadata(**entry["metadata"]),
            )
            for entry in raw
        ]
