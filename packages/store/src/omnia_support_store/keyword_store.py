"""Abstract keyword store interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Chunk, ScoredChunk, WhereFilter


class KeywordStore(ABC):
    """Interface for a keyword search store."""

    @abstractmethod
    def index(self, records: list[Chunk]) -> None:
        """Build the keyword index from records."""

    @abstractmethod
    def query(
        self,
        query_text: str,
        n_results: int,
        where: WhereFilter | None = None,
    ) -> list[ScoredChunk]:
        """Return the most relevant records for the query text."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all indexed data."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of indexed records."""
