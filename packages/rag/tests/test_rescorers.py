"""Tests for concrete rescorer implementations."""

from __future__ import annotations

from omnia_support_rag.query_interpretation import DocumentTypeRelevance, InterpretedQuery
from omnia_support_rag.rescorers import DocumentTypeRescorer, ResolutionStatusRescorer, SatisfactionRescorer
from omnia_support_store.models import ChunkMetadata, ScoredChunk


def _make_chunk(
    *,
    score: float = 1.0,
    customer_satisfaction: float | None = None,
    resolution_status: str | None = None,
) -> ScoredChunk:
    metadata: ChunkMetadata = {
        "document_id": "doc-1",
        "type": "support_ticket",
        "category": "General",
        "date": "2025-01-01",
        "product_category": "laptops",
        "chunk_index": 0,
        "total_chunks": 1,
    }
    if customer_satisfaction is not None:
        metadata["customer_satisfaction"] = customer_satisfaction
    if resolution_status is not None:
        metadata["resolution_status"] = resolution_status
    return ScoredChunk(id="doc-1-chunk-0", chunk="text", metadata=metadata, score=score)


class TestSatisfactionRescorer:
    def test_five_stars_boosts_score(self):
        chunk = _make_chunk(score=1.0, customer_satisfaction=5.0)
        rescorer = SatisfactionRescorer()

        result = rescorer.rescore(chunk)

        assert result.score == 1.0 * (1.0 + 0.2 * (5.0 - 3.0) / 2.0)
        assert result.score > chunk.score

    def test_one_star_reduces_score(self):
        chunk = _make_chunk(score=1.0, customer_satisfaction=1.0)
        rescorer = SatisfactionRescorer()

        result = rescorer.rescore(chunk)

        assert result.score == 1.0 * (1.0 + 0.2 * (1.0 - 3.0) / 2.0)
        assert result.score < chunk.score

    def test_three_stars_is_neutral(self):
        chunk = _make_chunk(score=1.0, customer_satisfaction=3.0)
        rescorer = SatisfactionRescorer()

        result = rescorer.rescore(chunk)

        assert result.score == chunk.score

    def test_missing_satisfaction_returns_unchanged(self):
        chunk = _make_chunk(score=1.0)
        rescorer = SatisfactionRescorer()

        result = rescorer.rescore(chunk)

        assert result is chunk

    def test_custom_weight(self):
        chunk = _make_chunk(score=1.0, customer_satisfaction=5.0)
        rescorer = SatisfactionRescorer(weight=0.5)

        result = rescorer.rescore(chunk)

        assert result.score == 1.0 * (1.0 + 0.5 * (5.0 - 3.0) / 2.0)


class TestResolutionStatusRescorer:
    def test_resolved_boosts_score(self):
        chunk = _make_chunk(score=1.0, resolution_status="resolved")
        rescorer = ResolutionStatusRescorer()

        result = rescorer.rescore(chunk)

        assert result.score == 1.2

    def test_active_is_neutral(self):
        chunk = _make_chunk(score=1.0, resolution_status="active")
        rescorer = ResolutionStatusRescorer()

        result = rescorer.rescore(chunk)

        assert result.score == chunk.score

    def test_pending_reduces_score(self):
        chunk = _make_chunk(score=1.0, resolution_status="pending")
        rescorer = ResolutionStatusRescorer()

        result = rescorer.rescore(chunk)

        assert result.score == 0.8

    def test_missing_status_returns_unchanged(self):
        chunk = _make_chunk(score=1.0)
        rescorer = ResolutionStatusRescorer()

        result = rescorer.rescore(chunk)

        assert result is chunk

    def test_custom_weights(self):
        chunk = _make_chunk(score=1.0, resolution_status="resolved")
        rescorer = ResolutionStatusRescorer(resolved_weight=2.0)

        result = rescorer.rescore(chunk)

        assert result.score == 2.0


def _make_intent(*, policy: int = 5, faq: int = 5, support_ticket: int = 5) -> InterpretedQuery:
    return InterpretedQuery(
        query="test",
        doc_type_relevance=DocumentTypeRelevance(policy=policy, faq=faq, support_ticket=support_ticket),
    )


class TestDocumentTypeRescorer:
    def test_high_relevance_boosts_score(self):
        intent = _make_intent(support_ticket=10)
        chunk = _make_chunk(score=1.0)
        rescorer = DocumentTypeRescorer()

        result = rescorer.rescore(chunk, intent)

        assert result.score == 1.0 * (1.0 + 0.3 * (10 - 5) / 5)
        assert result.score > chunk.score

    def test_low_relevance_reduces_score(self):
        intent = _make_intent(support_ticket=0)
        chunk = _make_chunk(score=1.0)
        rescorer = DocumentTypeRescorer()

        result = rescorer.rescore(chunk, intent)

        assert result.score == 1.0 * (1.0 + 0.3 * (0 - 5) / 5)
        assert result.score < chunk.score

    def test_neutral_relevance_unchanged(self):
        intent = _make_intent(support_ticket=5)
        chunk = _make_chunk(score=1.0)
        rescorer = DocumentTypeRescorer()

        result = rescorer.rescore(chunk, intent)

        assert result.score == chunk.score

    def test_no_intent_returns_unchanged(self):
        chunk = _make_chunk(score=1.0)
        rescorer = DocumentTypeRescorer()

        result = rescorer.rescore(chunk)

        assert result is chunk

    def test_missing_type_returns_unchanged(self):
        intent = _make_intent(policy=10, faq=10, support_ticket=10)
        metadata: ChunkMetadata = {  # ty: ignore[missing-typed-dict-key]
            "document_id": "doc-1",
            "category": "General",
            "date": "2025-01-01",
            "product_category": "laptops",
            "chunk_index": 0,
            "total_chunks": 1,
        }
        chunk = ScoredChunk(id="doc-1-chunk-0", chunk="text", metadata=metadata, score=1.0)
        rescorer = DocumentTypeRescorer()

        result = rescorer.rescore(chunk, intent)

        assert result is chunk

    def test_unknown_type_returns_unchanged(self):
        intent = _make_intent(policy=10, faq=10, support_ticket=10)
        metadata: ChunkMetadata = {
            "document_id": "doc-1",
            "type": "unknown_type",
            "category": "General",
            "date": "2025-01-01",
            "product_category": "laptops",
            "chunk_index": 0,
            "total_chunks": 1,
        }
        chunk = ScoredChunk(id="doc-1-chunk-0", chunk="text", metadata=metadata, score=1.0)
        rescorer = DocumentTypeRescorer()

        result = rescorer.rescore(chunk, intent)

        assert result is chunk

    def test_custom_weight(self):
        intent = _make_intent(support_ticket=10)
        chunk = _make_chunk(score=1.0)
        rescorer = DocumentTypeRescorer(weight=0.5)

        result = rescorer.rescore(chunk, intent)

        assert result.score == 1.0 * (1.0 + 0.5 * (10 - 5) / 5)
