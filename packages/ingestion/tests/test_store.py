"""Tests for the store module."""

from unittest.mock import MagicMock, patch

from chromadb.errors import NotFoundError

from omnia_support_ingestion.store import (
    COLLECTION_NAME,
    ChromaRecord,
    get_collection,
    should_skip_ingestion,
    upload_records,
)


class TestShouldSkipIngestion:
    def _mock_client(self, *, count: int) -> MagicMock:
        collection = MagicMock()
        collection.count.return_value = count
        client = MagicMock()
        client.get_collection.return_value = collection
        return client

    def test_skip_when_collection_has_documents(self):
        client = self._mock_client(count=42)
        assert should_skip_ingestion(client, force=False) is True

    def test_no_skip_when_force_is_true(self):
        client = self._mock_client(count=42)
        assert should_skip_ingestion(client, force=True) is False

    def test_no_skip_when_collection_not_found(self):
        client = MagicMock()
        client.get_collection.side_effect = NotFoundError("")
        assert should_skip_ingestion(client, force=False) is False

    def test_no_skip_when_collection_is_empty(self):
        client = self._mock_client(count=0)
        assert should_skip_ingestion(client, force=False) is False

    @patch("builtins.print")
    def test_prints_message_when_skipping(self, mock_print: MagicMock):
        client = self._mock_client(count=10)
        should_skip_ingestion(client, force=False)
        mock_print.assert_called_once()
        assert COLLECTION_NAME in mock_print.call_args[0][0]


class TestGetCollection:
    def test_creates_collection(self):
        client = MagicMock()
        embedding_fn = MagicMock()
        collection = MagicMock()
        client.get_or_create_collection.return_value = collection

        result = get_collection(client, embedding_fn)

        assert result is collection
        client.get_or_create_collection.assert_called_once_with(name=COLLECTION_NAME, embedding_function=embedding_fn)
        client.delete_collection.assert_not_called()

    def test_deletes_existing_when_should_delete(self):
        client = MagicMock()
        embedding_fn = MagicMock()
        client.get_or_create_collection.return_value = MagicMock()

        get_collection(client, embedding_fn, should_delete=True)

        client.delete_collection.assert_called_once_with(COLLECTION_NAME)

    def test_delete_tolerates_missing_collection(self):
        client = MagicMock()
        client.delete_collection.side_effect = NotFoundError("")
        client.get_or_create_collection.return_value = MagicMock()

        result = get_collection(client, MagicMock(), should_delete=True)

        assert result is not None


class TestUploadRecords:
    def _make_records(self, n: int = 2) -> list[ChromaRecord]:
        return [
            ChromaRecord(
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

    def test_upserts_all_records(self):
        collection = MagicMock()
        records = self._make_records(3)

        upload_records(collection, records)

        collection.upsert.assert_called_once()
        call_kwargs = collection.upsert.call_args
        assert call_kwargs[1]["ids"] == ["id-0", "id-1", "id-2"]

    @patch("omnia_support_ingestion.store.upload_records.retry.sleep")
    def test_retries_on_failure(self, mock_sleep: MagicMock):  # noqa: ARG002
        collection = MagicMock()
        collection.upsert.side_effect = [ConnectionError("down"), None]
        records = self._make_records()

        upload_records(collection, records)

        assert collection.upsert.call_count == 2
