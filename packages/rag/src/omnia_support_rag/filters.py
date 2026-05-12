"""Post-retrieval filters that operate on raw ScoredChunk results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from omnia_support_store import ScoredChunk


class RetrievalFilter(Protocol):
    """Protocol for a post-retrieval filter on raw store results."""

    def filter(self, results: list[ScoredChunk]) -> list[ScoredChunk]:
        """Return a filtered subset of results."""
        ...


class CosineSimThresholdFilter:
    """Drop results below a cosine similarity threshold.

    Default of 0.27 chosen empirically: relevant queries scored >= 0.31
    cosine similarity while out-of-scope queries scored <= 0.23, placing
    the threshold in the middle of the observed gap.
    """

    def __init__(self, min_similarity: float = 0.27) -> None:
        """Create a cosine similarity threshold filter."""
        self._min_similarity = min_similarity

    def filter(self, results: list[ScoredChunk]) -> list[ScoredChunk]:  # noqa: D102
        return [r for r in results if r.score >= self._min_similarity]


class BM25ThresholdFilter:
    """Drop results below a minimum BM25 score.

    Default of 1.5 chosen empirically: relevant queries produced top
    BM25 scores ≥ 1.66 while out-of-scope queries with zero term overlap
    scored 0.0. Vocabulary-overlap false positives (e.g. "stock market")
    scored up to ~1.7, so this threshold catches only the clearest misses.
    """

    def __init__(self, min_score: float = 1.5) -> None:
        """Create a BM25 score threshold filter."""
        self._min_score = min_score

    def filter(self, results: list[ScoredChunk]) -> list[ScoredChunk]:  # noqa: D102
        return [r for r in results if r.score >= self._min_score]
