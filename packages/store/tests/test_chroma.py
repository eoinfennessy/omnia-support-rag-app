"""Tests for the ChromaDB vector store implementation."""

from unittest.mock import MagicMock, patch

from chromadb.errors import NotFoundError

from omnia_support_store import DEFAULT_COLLECTION_NAME, ChromaVectorStore


class TestChromaVectorStore:
    @patch("chromadb.HttpClient")
    def test_init_creates_collection(self, mock_http_client_cls: MagicMock):
        mock_client = MagicMock()
        mock_http_client_cls.return_value = mock_client
        mock_client.get_or_create_collection.return_value = MagicMock()
        embedding_fn = MagicMock()

        ChromaVectorStore(host="localhost", port=8000, embedding_fn=embedding_fn)

        mock_client.get_or_create_collection.assert_called_once_with(
            name=DEFAULT_COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
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
    def test_query_passes_where_filter(self, mock_http_client_cls: MagicMock):
        mock_client = MagicMock()
        mock_http_client_cls.return_value = mock_client
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["1", "2"]],
            "documents": [["Returns policy info", "Shipping policy details"]],
            "metadatas": [[{"type": "policy"}, {"type": "policy"}]],
            "distances": [[0.1, 0.3]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection

        store = ChromaVectorStore(host="localhost", port=8000, embedding_fn=MagicMock())
        where = {"type": "policy"}
        results = store.query("return policy", 5, where=where)

        mock_collection.query.assert_called_once_with(
            query_texts=["return policy"],
            n_results=5,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        assert len(results) == 2
        assert all(r.metadata["type"] == "policy" for r in results)

    @patch("chromadb.HttpClient")
    def test_reset_tolerates_missing_collection(self, mock_http_client_cls: MagicMock):
        mock_client = MagicMock()
        mock_http_client_cls.return_value = mock_client
        mock_client.get_or_create_collection.return_value = MagicMock()
        mock_client.delete_collection.side_effect = NotFoundError("")

        store = ChromaVectorStore(host="localhost", port=8000, embedding_fn=MagicMock())
        store.reset()

        assert mock_client.get_or_create_collection.call_count == 2

    @patch("chromadb.HttpClient")
    def test_collection_property_returns_underlying_collection(self, mock_http_client_cls: MagicMock):
        mock_client = MagicMock()
        mock_http_client_cls.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        store = ChromaVectorStore(host="localhost", port=8000, embedding_fn=MagicMock())

        assert store.collection is mock_collection
