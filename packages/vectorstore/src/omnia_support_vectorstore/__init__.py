"""ChromaDB vector store layer for Omnia Support."""

from omnia_support_vectorstore.base import VectorStore
from omnia_support_vectorstore.chromadb import DEFAULT_COLLECTION_NAME, ChromaVectorStore
from omnia_support_vectorstore.models import (
    ScoredVectorRecord,
    VectorRecord,
    VectorRecordMetadata,
)

__all__ = [
    "DEFAULT_COLLECTION_NAME",
    "ChromaVectorStore",
    "ScoredVectorRecord",
    "VectorRecord",
    "VectorRecordMetadata",
    "VectorStore",
]
