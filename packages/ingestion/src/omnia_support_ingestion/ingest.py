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
from omnia_support_vectorstore import (
    ChromaVectorStore,
    VectorRecord,
)

from .chunk import build_embed_text, chunk_document

if TYPE_CHECKING:
    from omnia_support_models import Document
    from omnia_support_vectorstore import VectorRecordMetadata, VectorStore


@dataclass(frozen=True)
class IngestConfig:
    """Configuration for the ingestion pipeline."""

    input_path: Path
    chroma_host: str
    chroma_port: int
    openai_api_key: str
    embedding_model: str
    force: bool


def load_dataset(path: Path) -> Dataset:
    """Read and validate the dataset JSON file."""
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return Dataset.model_validate(raw)


def should_skip_ingestion(store: VectorStore, *, force: bool) -> bool:
    """Check whether ingestion should be skipped."""
    if force:
        return False
    count = store.count()
    if not count:
        return False
    print(f"Collection '{store.collection_name}' already has documents. Skipping ingestion. Use --force to re-ingest.")
    return True


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def _upload_records(store: VectorStore, records: list[VectorRecord]) -> None:
    """Upload records to ChromaDB with retry."""
    store.upload(records)


def _build_record_metadata(doc: Document, chunk_index: int, total_chunks: int) -> VectorRecordMetadata:
    """Build ChromaDB metadata for a record, omitting None fields."""
    metadata: VectorRecordMetadata = {
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


def build_records(documents: list[Document]) -> list[VectorRecord]:
    """Build ChromaDB records from documents.

    Chunks long documents and assigns unique IDs per chunk.
    """
    result: list[VectorRecord] = []

    for doc in documents:
        text_chunks = chunk_document(doc)
        total_chunks = len(text_chunks)

        result.extend(
            VectorRecord(
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

    dataset = load_dataset(input_path)
    records = build_records(dataset.documents)
    print(json.dumps([asdict(r) for r in records], indent=2))


def ingest(config: IngestConfig) -> None:
    """Load documents from JSON and ingest into ChromaDB."""
    if not config.input_path.exists():
        raise SystemExit(f"Error: input file not found: {config.input_path}")

    print(f"Connecting to ChromaDB at {config.chroma_host}:{config.chroma_port}...")
    store = ChromaVectorStore(
        host=config.chroma_host,
        port=config.chroma_port,
        embedding_fn=OpenAIEmbeddingFunction(api_key=config.openai_api_key, model_name=config.embedding_model),  # ty: ignore[invalid-argument-type]
    )

    if should_skip_ingestion(store, force=config.force):
        return

    if config.force:
        store.reset()
        print(f"  Deleted existing '{store.collection_name}' collection")

    print(f"Reading {config.input_path}...")
    dataset = load_dataset(config.input_path)
    print(f"  Loaded {len(dataset.documents)} documents")

    print("Preparing documents...")
    records = build_records(dataset.documents)
    print(f"  Prepared {len(records)} records")

    _upload_records(store, records)

    print(f"\nDone! Collection '{store.collection_name}' has {store.count()} documents.")
    print(f"  ChromaDB server: {config.chroma_host}:{config.chroma_port}")


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

    ingest(
        IngestConfig(
            input_path=Path(args.input),
            chroma_host=args.chroma_host,
            chroma_port=args.chroma_port,
            openai_api_key=openai_api_key,
            embedding_model=args.embedding_model,
            force=args.force,
        )
    )
