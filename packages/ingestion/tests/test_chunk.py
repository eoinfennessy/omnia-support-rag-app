"""Tests for the chunking module."""

import datetime

import pytest

from omnia_support_ingestion.chunk import (
    build_embed_text,
    chunk_document,
    chunk_text,
)
from omnia_support_models import Document, DocumentType, ProductCategory


def _make_doc(
    *,
    doc_type: DocumentType = DocumentType.SUPPORT_TICKET,
    content: str = "",
    category: str = "test",
    product_category: ProductCategory = ProductCategory.LAPTOPS,
) -> Document:
    return Document(
        document_id="DOC-001",
        type=doc_type,
        category=category,
        date=datetime.date(2025, 1, 1),
        resolution_status=None,
        customer_satisfaction=None,
        product_category=product_category,
        priority=None,
        content=content,
    )


class TestChunkText:
    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []

    def test_short_text_returns_single_chunk(self):
        result = chunk_text("Hello world", max_chars=100)
        assert result == ["Hello world"]

    def test_no_whitespace_still_terminates(self):
        text = "x" * 5000
        chunks = chunk_text(text, max_chars=2000, overlap=200)
        assert len(chunks) >= 3
        assert "".join(chunks) != text  # overlap causes duplication, but it terminates

    def test_splits_at_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, max_chars=35, overlap=0)
        assert len(chunks) >= 2
        assert chunks[0].endswith("Second sentence. ")

    def test_respects_max_chars(self):
        text = "word " * 500
        chunks = chunk_text(text, max_chars=100, overlap=0)
        assert all(len(c) <= 100 for c in chunks)

    def test_overlap_produces_repeated_content(self):
        text = "A " * 200
        chunks = chunk_text(text, max_chars=100, overlap=20)
        assert len(chunks) >= 3
        assert chunks[0][-20:] in chunks[1]


class TestChunkDocument:
    def test_faq_returns_whole_content(self):
        doc = _make_doc(doc_type=DocumentType.FAQ, content="Q: What?\nA: This.")
        assert chunk_document(doc) == ["Q: What?\nA: This."]

    def test_policy_without_header_passes_through(self):
        content = "Section One\nDetails.\n\nSection Two\nMore."
        doc = _make_doc(doc_type=DocumentType.POLICY, content=content)
        chunks = chunk_document(doc)
        assert len(chunks) == 2
        assert chunks[0].startswith("Section One")

    def test_policy_splits_on_sections(self):
        content = (
            "OMNIA RETAIL LTD — Returns Policy\n"
            "Effective Date: 2025-01-01\n"
            "Section One\nDetails here.\n\n"
            "Section Two\nMore details."
        )
        doc = _make_doc(doc_type=DocumentType.POLICY, content=content)
        chunks = chunk_document(doc)
        assert len(chunks) == 2
        assert chunks[0].startswith("Section One")
        assert chunks[1].startswith("Section Two")

    def test_ticket_strips_customer_line(self):
        content = "Customer: John Doe\nOrder: ORD-123 — Widget\nSome issue."
        doc = _make_doc(content=content)
        chunks = chunk_document(doc)
        assert not any("Customer:" in c for c in chunks)

    def test_ticket_replaces_order_prefix(self):
        content = "Order: ORD-456 — Laptop Pro\nScreen flickering."
        doc = _make_doc(content=content)
        chunks = chunk_document(doc)
        assert "Product: Laptop Pro" in chunks[0]

    def test_ticket_without_noise_fields_passes_through(self):
        content = "Just a plain ticket body with no structured headers."
        doc = _make_doc(content=content)
        chunks = chunk_document(doc)
        assert chunks == [content]


class TestBuildEmbedText:
    def test_includes_type_and_category(self):
        doc = _make_doc(category="Returns")
        result = build_embed_text(doc, "some chunk")
        assert "Type: support_ticket" in result
        assert "Category: Returns" in result

    def test_includes_product_category(self):
        doc = _make_doc(product_category=ProductCategory.MONITORS)
        result = build_embed_text(doc, "chunk")
        assert "Product: monitors" in result

    @pytest.mark.parametrize("pc", [ProductCategory.ALL, ProductCategory.NONE])
    def test_excludes_generic_product_categories(self, pc: ProductCategory):
        doc = _make_doc(product_category=pc)
        result = build_embed_text(doc, "chunk")
        assert "Product:" not in result
