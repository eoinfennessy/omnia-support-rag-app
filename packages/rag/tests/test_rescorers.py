"""Tests for concrete rescorer implementations."""

from __future__ import annotations

from omnia_support_rag.rescorers import ResolutionStatusRescorer, SatisfactionRescorer
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
