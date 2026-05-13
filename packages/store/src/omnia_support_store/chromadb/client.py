"""ChromaDB vector store implementation."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, cast

from chromadb.errors import NotFoundError

from ..models import ChunkMetadata, ScoredChunk
from ..vector_store import VectorStore

if TYPE_CHECKING:
    from chromadb import ClientAPI, Collection
    from chromadb.api.types import Embeddable, EmbeddingFunction

    from ..models import Chunk, WhereFilter

DEFAULT_COLLECTION_NAME = "omnia_support"


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        embedding_fn: EmbeddingFunction[Embeddable],
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """Connect to ChromaDB and initialise the collection."""
        import chromadb

        self._client: ClientAPI = chromadb.HttpClient(host=host, port=port)
        self._embedding_fn = embedding_fn
        self._collection_name = collection_name
        self._collection: Collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self) -> Collection:
        """Return the underlying ChromaDB collection."""
        return self._collection

    @property
    def collection_name(self) -> str:  # noqa: D102
        return self._collection_name

    def upload(self, records: list[Chunk]) -> None:  # noqa: D102 - docstring on ABC
        self._collection.upsert(
            ids=[r.id for r in records],
            documents=[r.chunk for r in records],
            metadatas=[r.metadata for r in records],  # ty: ignore[invalid-argument-type]
        )

    def query(  # noqa: D102 - docstring on ABC
        self,
        query_text: str,
        n_results: int,
        where: WhereFilter | None = None,
    ) -> list[ScoredChunk]:
        results = self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["metadatas"] or not results["distances"]:
            return []

        return [
            ScoredChunk(id=id_, chunk=doc, metadata=cast("ChunkMetadata", meta), score=1 - dist)
            for id_, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                strict=True,
            )
        ]

    def reset(self) -> None:  # noqa: D102
        with contextlib.suppress(NotFoundError):
            self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_fn,
        )

    def count(self) -> int:  # noqa: D102 - docstring on ABC
        return self._collection.count()
