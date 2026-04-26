"""Pydantic models and enums for the Omnia Retail dataset."""

import datetime  # noqa: TC003 — Pydantic needs this at runtime
import json
from enum import StrEnum
from pathlib import Path  # noqa: TC003 — used at runtime in Dataset.from_json

from pydantic import BaseModel, Field


class DocumentType(StrEnum):
    """Document type in the knowledge base."""

    SUPPORT_TICKET = "support_ticket"
    POLICY = "policy"
    FAQ = "faq"


class ResolutionStatus(StrEnum):
    """Resolution status of a support ticket."""

    RESOLVED = "resolved"
    ACTIVE = "active"
    PENDING = "pending"


class ProductCategory(StrEnum):
    """Product category for a document."""

    LAPTOPS = "laptops"
    MONITORS = "monitors"
    ACCESSORIES = "accessories"
    STORAGE = "storage"
    ALL = "all"
    NONE = "none"


class Priority(StrEnum):
    """Priority level of a support ticket."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DocumentTypes(BaseModel):
    """Count of documents by type."""

    support_tickets: int
    policies: int
    faqs: int


class DatasetMetadata(BaseModel):
    """Top-level metadata for the dataset."""

    name: str
    version: str
    created: datetime.date
    description: str
    total_documents: int
    document_types: DocumentTypes
    product_categories: list[ProductCategory]
    geographic_context: str
    currency: str
    regulatory_context: list[str]


class Document(BaseModel):
    """A single document in the knowledge base."""

    document_id: str
    type: DocumentType
    category: str
    date: datetime.date
    resolution_status: ResolutionStatus | None
    customer_satisfaction: float | None = Field(default=None, ge=0.0, le=5.0)
    product_category: ProductCategory
    priority: Priority | None
    content: str


class TestQuery(BaseModel):
    """A test query with expected retrieval results."""

    query_id: str
    query: str
    expected_document_ids: list[str]
    category: str
    notes: str


class Dataset(BaseModel):
    """Complete dataset with metadata, documents, and test queries."""

    dataset_metadata: DatasetMetadata
    documents: list[Document]
    test_queries: list[TestQuery]

    @classmethod
    def from_json(cls, path: Path) -> Dataset:
        """Read and validate a dataset JSON file."""
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
        return cls.model_validate(raw)
