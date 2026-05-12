"""Context formatters: token budget management and lost-in-the-middle reordering."""

from __future__ import annotations

from typing import Protocol

import mlflow
import tiktoken


class ContextFormatter(Protocol):
    """Protocol for a context formatting step applied after retrieval."""

    def format(self, results: list[dict]) -> list[dict]:
        """Return a transformed list of retrieval results."""
        ...


class TokenBudgetFormatter:
    """Trim results to fit within a token budget.

    Walks results in order (best-first from the retriever), accumulating
    tokens until the next result would exceed the budget.
    """

    def __init__(self, *, max_tokens: int = 3000) -> None:
        """Create a token budget formatter."""
        self._max_tokens = max_tokens
        self._encoding = tiktoken.get_encoding("cl100k_base")

    @mlflow.trace(name="token_budget_format")
    def format(self, results: list[dict]) -> list[dict]:  # noqa: D102
        selected: list[dict] = []
        token_count = 0
        for result in results:
            tokens = len(self._encoding.encode(result["page_content"]))
            if token_count + tokens > self._max_tokens:
                break
            selected.append(result)
            token_count += tokens
        return selected


class LostInMiddleFormatter:
    """Reorder results so the most relevant sit at the start and end.

    LLMs attend most to the beginning and end of their context window
    (Liu et al., 2023). Given results sorted best-first, this interleaves
    them so odd-ranked items fill from the front and even-ranked items
    fill from the back: [r1, r3, r5, r4, r2].
    """

    @mlflow.trace(name="lost_in_middle_reorder")
    def format(self, results: list[dict]) -> list[dict]:  # noqa: D102
        if len(results) <= 2:
            return list(results)
        reordered: list[dict] = [{}] * len(results)
        left, right = 0, len(results) - 1
        for i, result in enumerate(results):
            if i % 2 == 0:
                reordered[left] = result
                left += 1
            else:
                reordered[right] = result
                right -= 1
        return reordered
