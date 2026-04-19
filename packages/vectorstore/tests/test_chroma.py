"""Tests for the ChromaDB vector store implementation."""

from unittest.mock import MagicMock, patch

from chromadb.errors import NotFoundError

from omnia_support_vectorstore import DEFAULT_COLLECTION_NAME, ChromaVectorStore


class TestChromaVectorStore:
    @patch("chromadb.HttpClient")
    def test_init_creates_collection(self, mock_http_client_cls: MagicMock):
        mock_client = MagicMock()
        mock_http_client_cls.return_value = mock_client
        mock_client.get_or_create_collection.return_value = MagicMock()
        embedding_fn = MagicMock()

        ChromaVectorStore(host="localhost", port=8000, embedding_fn=embedding_fn)

        mock_client.get_or_create_collection.assert_called_once_with(
            name=DEFAULT_COLLECTION_NAME, embedding_function=embedding_fn
        )

    @patch("chromadb.HttpClient")
    def test_reset_deletes_and_recreates(self, mock_http_client_cls: MagicMock):
        mock_client = MagicMock()
        mock_http_client_cls.return_value = mock_client
        mock_client.get_or_create_collection.return_value = MagicMock()

        store = ChromaVectorStore(host="localhost", port=8000, embedding_fn=MagicMock())
        store.reset()

        mock_client.delete_collection.assert_called_once_with(DEFAULT_COLLECTION_NAME)
        assert mock_client.get_or_create_collection.call_count == 2

    @patch("chromadb.HttpClient")
    def test_reset_tolerates_missing_collection(self, mock_http_client_cls: MagicMock):
        mock_client = MagicMock()
        mock_http_client_cls.return_value = mock_client
        mock_client.get_or_create_collection.return_value = MagicMock()
        mock_client.delete_collection.side_effect = NotFoundError("")

        store = ChromaVectorStore(host="localhost", port=8000, embedding_fn=MagicMock())
        store.reset()

        assert mock_client.get_or_create_collection.call_count == 2
