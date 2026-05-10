"""Abstract vector store interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import ScoredVectorRecord, VectorRecord


class VectorStore(ABC):
    """Interface for a vector store backend."""

    @abstractmethod
    def upload(self, records: list[VectorRecord]) -> None:
        """Upload records to the store, replacing any with matching IDs."""

    @abstractmethod
    def query(
        self,
        query_text: str,
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> list[ScoredVectorRecord]:
        """Return the closest records to the query text."""

    @abstractmethod
    def reset(self) -> None:
        """Delete all data and recreate the collection."""

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Return the name of the backing collection."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of records in the store."""
