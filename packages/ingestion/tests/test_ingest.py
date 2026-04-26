"""Tests for record building in the ingest module."""

import datetime
import json
from typing import TYPE_CHECKING

import pytest

from omnia_support_ingestion.ingest import build_records

if TYPE_CHECKING:
    from pathlib import Path
from omnia_support_models import (
    Dataset,
    Document,
    DocumentType,
    Priority,
    ProductCategory,
    ResolutionStatus,
)


def _make_doc(
    *,
    document_id: str = "DOC-001",
    doc_type: DocumentType = DocumentType.FAQ,
    content: str = "Q: What?\nA: Answer.",
    resolution_status: ResolutionStatus | None = None,
    customer_satisfaction: float | None = None,
    priority: Priority | None = None,
) -> Document:
    return Document(
        document_id=document_id,
        type=doc_type,
        category="General",
        date=datetime.date(2025, 6, 1),
        resolution_status=resolution_status,
        customer_satisfaction=customer_satisfaction,
        product_category=ProductCategory.LAPTOPS,
        priority=priority,
        content=content,
    )


class TestBuildRecords:
    def test_single_faq_produces_one_record(self):
        records = build_records([_make_doc()])
        assert len(records) == 1
        assert records[0].id == "DOC-001-chunk-0"

    def test_record_metadata_has_required_fields(self):
        records = build_records([_make_doc()])
        meta = records[0].metadata
        assert meta["document_id"] == "DOC-001"
        assert meta["type"] == "faq"
        assert meta["chunk_index"] == 0
        assert meta["total_chunks"] == 1

    def test_optional_metadata_included_when_present(self):
        doc = _make_doc(
            resolution_status=ResolutionStatus.RESOLVED,
            customer_satisfaction=4.5,
            priority=Priority.HIGH,
        )
        records = build_records([doc])
        meta = records[0].metadata
        assert meta["resolution_status"] == "resolved"
        assert meta["customer_satisfaction"] == 4.5
        assert meta["priority"] == "high"

    def test_optional_metadata_absent_when_none(self):
        records = build_records([_make_doc()])
        meta = records[0].metadata
        assert "resolution_status" not in meta
        assert "customer_satisfaction" not in meta
        assert "priority" not in meta

    def test_multiple_documents(self):
        docs = [_make_doc(document_id=f"DOC-{i}") for i in range(3)]
        records = build_records(docs)
        assert len(records) == 3
        ids = {r.id for r in records}
        assert ids == {"DOC-0-chunk-0", "DOC-1-chunk-0", "DOC-2-chunk-0"}

    def test_long_ticket_produces_multiple_chunks(self):
        long_content = "Some issue description. " * 200
        doc = _make_doc(doc_type=DocumentType.SUPPORT_TICKET, content=long_content)
        records = build_records([doc])
        assert len(records) > 1
        assert all(r.metadata["total_chunks"] == len(records) for r in records)
        assert [r.metadata["chunk_index"] for r in records] == list(range(len(records)))

    def test_empty_document_list(self):
        assert build_records([]) == []

    def test_date_serialised_as_isoformat(self):
        records = build_records([_make_doc()])
        assert records[0].metadata["date"] == "2025-06-01"


_MINIMAL_DATASET = {
    "dataset_metadata": {
        "name": "test",
        "version": "0.1",
        "created": "2025-01-01",
        "description": "test dataset",
        "total_documents": 1,
        "document_types": {"support_tickets": 0, "policies": 0, "faqs": 1},
        "product_categories": ["laptops"],
        "geographic_context": "IE",
        "currency": "EUR",
        "regulatory_context": [],
    },
    "documents": [
        {
            "document_id": "DOC-001",
            "type": "faq",
            "category": "General",
            "date": "2025-01-01",
            "resolution_status": None,
            "customer_satisfaction": None,
            "product_category": "laptops",
            "priority": None,
            "content": "Q: Test?\nA: Yes.",
        },
    ],
    "test_queries": [],
}


class TestLoadDataset:
    def test_loads_and_validates(self, tmp_path: Path):
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps(_MINIMAL_DATASET))
        dataset = Dataset.from_json(path)
        assert len(dataset.documents) == 1
        assert dataset.documents[0].document_id == "DOC-001"

    def test_invalid_json_raises(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        with pytest.raises(json.JSONDecodeError):
            Dataset.from_json(path)
