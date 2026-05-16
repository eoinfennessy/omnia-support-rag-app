"""Tests for ingestion-specific store operations."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

from omnia_support_ingestion.ingest import IngestConfig, _should_skip_ingestion, _upload_records
from omnia_support_store import Chunk, KeywordStore, VectorStore


class TestShouldSkipIngestion:
    def _make_config(self, *, vector_count: int, keyword_count: int, force: bool = False) -> IngestConfig:
        vector_store = MagicMock(spec=VectorStore)
        vector_store.count.return_value = vector_count
        type(vector_store).collection_name = PropertyMock(return_value="test_collection")
        keyword_store = MagicMock(spec=KeywordStore)
        keyword_store.count.return_value = keyword_count
        return IngestConfig(
            input_path=Path("dummy.json"),
            vector_store=vector_store,
            keyword_store=keyword_store,
            force=force,
        )

    def test_skip_when_both_stores_have_data(self):
        config = self._make_config(vector_count=42, keyword_count=42)
        assert _should_skip_ingestion(config) is True

    def test_no_skip_when_force_is_true(self):
        config = self._make_config(vector_count=42, keyword_count=42, force=True)
        assert _should_skip_ingestion(config) is False

    def test_no_skip_when_vector_store_is_empty(self):
        config = self._make_config(vector_count=0, keyword_count=42)
        assert _should_skip_ingestion(config) is False

    def test_no_skip_when_keyword_store_is_empty(self):
        config = self._make_config(vector_count=42, keyword_count=0)
        assert _should_skip_ingestion(config) is False

    def test_no_skip_when_both_stores_are_empty(self):
        config = self._make_config(vector_count=0, keyword_count=0)
        assert _should_skip_ingestion(config) is False

    @patch("builtins.print")
    def test_prints_message_when_skipping(self, mock_print: MagicMock):
        config = self._make_config(vector_count=10, keyword_count=10)
        _should_skip_ingestion(config)
        mock_print.assert_called_once()


class TestUploadRecords:
    def _make_records(self, n: int = 2) -> list[Chunk]:
        return [
            Chunk(
                id=f"id-{i}",
                page_content=f"chunk {i}",
                metadata={
                    "document_id": f"doc-{i}",
                    "type": "faq",
                    "category": "test",
                    "date": "2025-01-01",
                    "product_category": "laptops",
                    "chunk_index": 0,
                    "total_chunks": 1,
                },
            )
            for i in range(n)
        ]

    def test_uploads_all_records(self):
        store = MagicMock(spec=VectorStore)
        records = self._make_records(3)

        _upload_records(store, records)

        store.upload.assert_called_once_with(records)

    @patch("omnia_support_ingestion.ingest._upload_records.retry.sleep")
    def test_retries_on_failure(self, mock_sleep: MagicMock):  # noqa: ARG002
        store = MagicMock(spec=VectorStore)
        store.upload.side_effect = [ConnectionError("down"), None]
        records = self._make_records()

        _upload_records(store, records)

        assert store.upload.call_count == 2
