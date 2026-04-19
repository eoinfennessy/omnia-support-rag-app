"""Tests for ingestion-specific store operations."""

from unittest.mock import MagicMock, PropertyMock, patch

from omnia_support_ingestion.ingest import _upload_records, should_skip_ingestion
from omnia_support_vectorstore import VectorRecord, VectorStore


class TestShouldSkipIngestion:
    def _mock_store(self, *, count: int, collection_name: str = "test_collection") -> MagicMock:
        store = MagicMock(spec=VectorStore)
        store.count.return_value = count
        type(store).collection_name = PropertyMock(return_value=collection_name)
        return store

    def test_skip_when_collection_has_documents(self):
        store = self._mock_store(count=42)
        assert should_skip_ingestion(store, force=False) is True

    def test_no_skip_when_force_is_true(self):
        store = self._mock_store(count=42)
        assert should_skip_ingestion(store, force=True) is False

    def test_no_skip_when_collection_is_empty(self):
        store = self._mock_store(count=0)
        assert should_skip_ingestion(store, force=False) is False

    @patch("builtins.print")
    def test_prints_message_when_skipping(self, mock_print: MagicMock):
        store = self._mock_store(count=10)
        should_skip_ingestion(store, force=False)
        mock_print.assert_called_once()
        assert "test_collection" in mock_print.call_args[0][0]


class TestUploadRecords:
    def _make_records(self, n: int = 2) -> list[VectorRecord]:
        return [
            VectorRecord(
                id=f"id-{i}",
                chunk=f"chunk {i}",
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
