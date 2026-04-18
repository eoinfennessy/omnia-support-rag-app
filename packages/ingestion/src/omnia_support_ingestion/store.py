"""ChromaDB storage layer for ingested chunks."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, NotRequired, TypedDict

from chromadb.errors import NotFoundError
from tenacity import retry, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from chromadb import ClientAPI, Collection
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

COLLECTION_NAME = "omnia_support"


class ChromaRecordMetadata(TypedDict):
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


class ChromaRecord(NamedTuple):
    """A chunk of text with its ID and metadata, ready for ChromaDB."""

    id: str
    chunk: str
    metadata: ChromaRecordMetadata


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def upload_records(collection: Collection, records: list[ChromaRecord]) -> None:
    """Upload records to ChromaDB."""
    collection.upsert(
        ids=[r.id for r in records],
        documents=[r.chunk for r in records],
        metadatas=[r.metadata for r in records],  # ty: ignore[invalid-argument-type]
    )


def should_skip_ingestion(client: ClientAPI, *, force: bool) -> bool:
    """Check whether ingestion should be skipped."""
    if force:
        return False
    try:
        count = client.get_collection(COLLECTION_NAME).count()
    except NotFoundError:
        return False
    if not count:
        return False
    print(f"Collection '{COLLECTION_NAME}' already has documents. Skipping ingestion. Use --force to re-ingest.")
    return True


def get_collection(
    client: ClientAPI, embedding_fn: OpenAIEmbeddingFunction, *, should_delete: bool = False
) -> Collection:
    """Set up the ChromaDB collection, deleting existing data if requested."""
    if should_delete:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"  Deleted existing '{COLLECTION_NAME}' collection")
        except NotFoundError:
            pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,  # ty: ignore[invalid-argument-type]
    )
