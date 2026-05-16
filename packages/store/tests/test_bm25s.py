"""Tests for the BM25 keyword store implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnia_support_store import BM25sKeywordStore, Chunk, ChunkMetadata

if TYPE_CHECKING:
    from pathlib import Path


def _make_chunks() -> list[Chunk]:
    return [
        Chunk(
            id="doc-1-chunk-0",
            page_content="The return policy allows refunds within 30 days of purchase",
            metadata=ChunkMetadata(
                document_id="doc-1",
                type="policy",
                category="Returns",
                date="2025-01-01",
                product_category="all",
                chunk_index=0,
                total_chunks=1,
            ),
        ),
        Chunk(
            id="doc-2-chunk-0",
            page_content="Shipping takes 3 to 5 business days for standard delivery",
            metadata=ChunkMetadata(
                document_id="doc-2",
                type="policy",
                category="Shipping",
                date="2025-01-01",
                product_category="all",
                chunk_index=0,
                total_chunks=1,
            ),
        ),
        Chunk(
            id="doc-3-chunk-0",
            page_content="How do I track my laptop order status and delivery date",
            metadata=ChunkMetadata(
                document_id="doc-3",
                type="faq",
                category="Orders",
                date="2025-01-01",
                product_category="laptops",
                chunk_index=0,
                total_chunks=1,
            ),
        ),
    ]


class TestBM25sKeywordStore:
    def test_index_and_query(self):
        store = BM25sKeywordStore()
        store.index(_make_chunks())

        results = store.query("return policy refund", 2)

        assert len(results) > 0
        assert results[0].id == "doc-1-chunk-0"
        assert results[0].metadata["document_id"] == "doc-1"
        assert results[0].score > 0

    def test_empty_index_returns_empty(self):
        store = BM25sKeywordStore()

        results = store.query("anything", 5)

        assert results == []

    def test_count(self):
        store = BM25sKeywordStore()
        assert store.count() == 0

        store.index(_make_chunks())
        assert store.count() == 3

    def test_reset_clears_index(self):
        store = BM25sKeywordStore()
        store.index(_make_chunks())
        assert store.count() == 3

        store.reset()

        assert store.count() == 0
        assert store.query("return", 5) == []

    def test_save_and_load(self, tmp_path: Path):
        store = BM25sKeywordStore(index_path=tmp_path / "bm25")
        store.index(_make_chunks())

        loaded = BM25sKeywordStore(index_path=tmp_path / "bm25")

        assert loaded.count() == 3
        results = loaded.query("return policy refund", 2)
        assert len(results) > 0
        assert results[0].id == "doc-1-chunk-0"

    def test_auto_loads_from_path(self, tmp_path: Path):
        index_path = tmp_path / "bm25"
        store = BM25sKeywordStore(index_path=index_path)
        store.index(_make_chunks())

        fresh = BM25sKeywordStore(index_path=index_path)

        assert fresh.count() == 3

    def test_query_k_exceeds_corpus(self):
        store = BM25sKeywordStore()
        store.index(_make_chunks())

        results = store.query("return", 100)

        assert len(results) <= 3

    def test_where_filter(self):
        store = BM25sKeywordStore()
        store.index(_make_chunks())

        results = store.query("delivery", 5, where={"type": "faq"})

        assert all(r.metadata["type"] == "faq" for r in results)

    def test_where_filter_in_operator(self):
        store = BM25sKeywordStore()
        store.index(_make_chunks())

        results = store.query("delivery policy", 5, where={"product_category": {"$in": ["laptops", "monitors"]}})

        assert all(r.metadata["product_category"] in ("laptops", "monitors") for r in results)
        assert len(results) > 0

    def test_where_filter_in_excludes_non_matching(self):
        store = BM25sKeywordStore()
        store.index(_make_chunks())

        results = store.query("return policy", 5, where={"product_category": {"$in": ["laptops"]}})

        assert all(r.metadata["product_category"] == "laptops" for r in results)
