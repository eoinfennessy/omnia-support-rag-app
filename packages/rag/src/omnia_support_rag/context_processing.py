"""Context processors: token budget management and lost-in-the-middle reordering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import mlflow
import tiktoken

if TYPE_CHECKING:
    from omnia_support_store import ScoredChunk


class ContextProcessor(Protocol):
    """Protocol for a post-retrieval processing step."""

    def process(self, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        """Return a transformed list of retrieval chunks."""
        ...


class TokenBudgetFormatter:
    """Trim chunks to fit within a token budget.

    Walks chunks in order (best-first from the retriever), accumulating
    tokens until the next result would exceed the budget.
    """

    def __init__(self, *, max_tokens: int = 3000) -> None:
        """Create a token budget formatter."""
        self._max_tokens = max_tokens
        self._encoding = tiktoken.get_encoding("cl100k_base")

    @mlflow.trace(name="token_budget_format")
    def process(self, chunks: list[ScoredChunk]) -> list[ScoredChunk]:  # noqa: D102
        selected: list[ScoredChunk] = []
        token_count = 0
        for chunk in chunks:
            tokens = len(self._encoding.encode(chunk.chunk))
            if token_count + tokens > self._max_tokens:
                break
            selected.append(chunk)
            token_count += tokens
        return selected


class LostInMiddleFormatter:
    """Reorder chunks so the most relevant sit at the start and end.

    LLMs attend most to the beginning and end of their context window
    (Liu et al., 2023). Given chunks sorted best-first, this interleaves
    them so odd-ranked items fill from the front and even-ranked items
    fill from the back: [r1, r3, r5, r4, r2].
    """

    @mlflow.trace(name="lost_in_middle_reorder")
    def process(self, chunks: list[ScoredChunk]) -> list[ScoredChunk]:  # noqa: D102
        if len(chunks) <= 2:
            return list(chunks)
        reordered = list(chunks)
        left, right = 0, len(chunks) - 1
        for i, chunk in enumerate(chunks):
            if i % 2 == 0:
                reordered[left] = chunk
                left += 1
            else:
                reordered[right] = chunk
                right -= 1
        return reordered
