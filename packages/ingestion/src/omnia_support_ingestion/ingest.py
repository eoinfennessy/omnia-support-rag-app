"""Ingestion pipeline for loading the Omnia Retail dataset into ChromaDB."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from omnia_support_models import Dataset
from omnia_support_store import (
    BM25sKeywordStore,
    ChromaVectorStore,
    Chunk,
)

from .chunk import build_embed_text, chunk_document

if TYPE_CHECKING:
    from omnia_support_models import Document
    from omnia_support_store import ChunkMetadata, KeywordStore, VectorStore


@dataclass(frozen=True)
class IngestConfig:
    """Configuration for the ingestion pipeline."""

    input_path: Path
    vector_store: VectorStore
    keyword_store: KeywordStore
    force: bool


def _should_skip_ingestion(config: IngestConfig) -> bool:
    """Check whether ingestion should be skipped."""
    if config.force:
        return False
    if not config.vector_store.count() or not config.keyword_store.count():
        return False
    print("Stores already populated. Skipping ingestion. Use --force to re-ingest.")
    return True


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def _upload_records(store: VectorStore, records: list[Chunk]) -> None:
    """Upload records to the vector store with retry."""
    store.upload(records)


def _build_record_metadata(doc: Document, chunk_index: int, total_chunks: int) -> ChunkMetadata:
    """Build ChromaDB metadata for a record, omitting None fields."""
    metadata: ChunkMetadata = {
        "document_id": doc.document_id,
        "type": doc.type.value,
        "category": doc.category,
        "date": doc.date.isoformat(),
        "product_category": doc.product_category.value,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
    }
    if doc.resolution_status is not None:
        metadata["resolution_status"] = doc.resolution_status.value
    if doc.customer_satisfaction is not None:
        metadata["customer_satisfaction"] = doc.customer_satisfaction
    if doc.priority is not None:
        metadata["priority"] = doc.priority.value
    return metadata


def build_records(documents: list[Document]) -> list[Chunk]:
    """Build ChromaDB records from documents.

    Chunks long documents and assigns unique IDs per chunk.
    """
    result: list[Chunk] = []

    for doc in documents:
        text_chunks = chunk_document(doc)
        total_chunks = len(text_chunks)

        result.extend(
            Chunk(
                id=f"{doc.document_id}-chunk-{i}",
                chunk=build_embed_text(doc, text),
                metadata=_build_record_metadata(doc, chunk_index=i, total_chunks=total_chunks),
            )
            for i, text in enumerate(text_chunks)
        )

    return result


def dry_run(input_path: Path) -> None:
    """Load and chunk documents, printing the result as JSON without ingesting."""
    if not input_path.exists():
        raise SystemExit(f"Error: input file not found: {input_path}")

    dataset = Dataset.from_json(input_path)
    records = build_records(dataset.documents)
    print(json.dumps([asdict(r) for r in records], indent=2))


def ingest(config: IngestConfig) -> None:
    """Load documents from JSON and ingest into ChromaDB."""
    if not config.input_path.exists():
        raise SystemExit(f"Error: input file not found: {config.input_path}")

    if _should_skip_ingestion(config):
        return

    if config.force:
        config.vector_store.reset()
        config.keyword_store.reset()
        print("  Reset vector store and keyword index")

    print(f"Reading {config.input_path}...")
    dataset = Dataset.from_json(config.input_path)
    print(f"  Loaded {len(dataset.documents)} documents")

    print("Preparing documents...")
    records = build_records(dataset.documents)
    print(f"  Prepared {len(records)} records")

    _upload_records(config.vector_store, records)
    print(f"  Uploaded {config.vector_store.count()} records to vector store")

    config.keyword_store.index(records)
    print(f"  Indexed {config.keyword_store.count()} records in keyword store")

    print("\nDone!")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest Omnia Retail documents into ChromaDB",
    )
    parser.add_argument(
        "--input",
        "-i",
        default="dataset/omnia_retail_knowledge_base.json",
        help="Path to dataset JSON file (default: dataset/omnia_retail_knowledge_base.json)",
    )
    parser.add_argument(
        "--chroma-host",
        default=os.environ.get("CHROMA_HOST", "localhost"),
        help="ChromaDB server host (default: $CHROMA_HOST or localhost)",
    )
    parser.add_argument(
        "--chroma-port",
        type=int,
        default=int(os.environ.get("CHROMA_PORT", "8000")),
        help="ChromaDB server port (default: $CHROMA_PORT or 8000)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=os.environ.get("FORCE_INGESTION", "").lower() == "true",
        help="Force re-ingestion even if data exists (default: $FORCE_INGESTION or false)",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        help="OpenAI embedding model (default: $EMBEDDING_MODEL or text-embedding-3-small)",
    )
    parser.add_argument(
        "--bm25-path",
        default=os.environ.get("BM25_INDEX_PATH", "data/bm25"),
        help="Path to save BM25 keyword index (default: $BM25_INDEX_PATH or data/bm25)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prepared chunks as JSON without ingesting",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for ingesting documents into ChromaDB."""
    load_dotenv()

    args = _parse_args()

    if args.dry_run:
        dry_run(Path(args.input))
        return

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise SystemExit("Error: OPENAI_API_KEY environment variable is required")

    embedding_fn = OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name=args.embedding_model,
    )

    vector_store = ChromaVectorStore(
        host=args.chroma_host,
        port=args.chroma_port,
        embedding_fn=embedding_fn,  # ty: ignore[invalid-argument-type]
    )

    keyword_store = BM25sKeywordStore(index_path=Path(args.bm25_path))

    ingest(
        IngestConfig(
            input_path=Path(args.input),
            vector_store=vector_store,
            keyword_store=keyword_store,
            force=args.force,
        )
    )
