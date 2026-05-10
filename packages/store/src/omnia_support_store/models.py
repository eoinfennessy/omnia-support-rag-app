"""Record types shared across vector and keyword stores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired, TypedDict


class ChunkMetadata(TypedDict):
    """Metadata stored alongside each chunk."""

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
class Chunk:
    """A chunk of text with its ID and metadata."""

    id: str
    chunk: str
    metadata: ChunkMetadata


@dataclass(frozen=True)
class ScoredChunk(Chunk):
    """A Chunk returned from a query, with a distance score."""

    distance: float = 0.0
