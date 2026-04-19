"""ChromaDB record types shared across ingestion and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired, TypedDict


class VectorRecordMetadata(TypedDict):
    """Metadata stored alongside each record in ChromaDB."""

    document_id: str
    type: str
    category: str
    date: str
    product_category: str
    chunk_index: int
    total_chunks: int
    resolution_status: NotRequired[str]
    customer_satisfaction: NotRequired[float]
    priority: NotRequired[str]


@dataclass(frozen=True)
class VectorRecord:
    """A chunk of text with its ID and metadata, ready for ChromaDB."""

    id: str
    chunk: str
    metadata: VectorRecordMetadata


@dataclass(frozen=True)
class ScoredVectorRecord(VectorRecord):
    """A VectorRecord returned from a query, with a distance score."""

    distance: float = 0.0
