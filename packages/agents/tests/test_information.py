"""Tests for the information agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from omnia_support_agents.information import InformationAgent
from omnia_support_agents.models import InformationResult
from omnia_support_rag.rag import RagResult
from omnia_support_store import ChunkMetadata, ScoredChunk

_STUB_METADATA: ChunkMetadata = {
    "document_id": "doc-1",
    "type": "faq",
    "category": "general",
    "date": "2024-01-01",
    "product_category": "all",
    "chunk_index": 0,
    "total_chunks": 1,
}


def _make_chunk(content: str, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(id="chunk-id", page_content=content, metadata=_STUB_METADATA, score=score)


def _make_agent(**overrides) -> InformationAgent:
    defaults = {
        "completions_client": MagicMock(),
        "retriever": MagicMock(),
        "prompt_uri": "prompts:/rag@production",
        "n_results": 5,
        "model": "gpt-4",
    }
    return InformationAgent(**{**defaults, **overrides})  # ty: ignore[invalid-argument-type]


class TestInformationAgent:
    @patch("omnia_support_agents.information.run_rag_pipeline")
    def test_returns_information_result(self, mock_rag):
        chunks = [_make_chunk("chunk1"), _make_chunk("chunk2")]
        mock_rag.return_value = RagResult(answer="the answer", retrieved_chunks=chunks)

        result = _make_agent().handle("What is the return policy?")

        assert isinstance(result, InformationResult)
        assert result.query == "What is the return policy?"
        assert result.answer == "the answer"
        assert result.context == chunks

    @patch("omnia_support_agents.information.run_rag_pipeline")
    def test_passes_history_to_rag_pipeline(self, mock_rag):
        mock_rag.return_value = RagResult(answer="answer", retrieved_chunks=[])
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        _make_agent().handle("follow-up question", history=history)  # ty: ignore[invalid-argument-type]

        assert mock_rag.call_args.kwargs["history"] is history

    @patch("omnia_support_agents.information.run_rag_pipeline")
    def test_passes_config_to_rag_pipeline(self, mock_rag):
        mock_rag.return_value = RagResult(answer="answer", retrieved_chunks=[])
        client = MagicMock()
        retriever = MagicMock()

        _make_agent(
            completions_client=client,
            retriever=retriever,
            prompt_uri="prompts:/custom@v1",
            n_results=10,
            model="gpt-4o",
        ).handle("question")

        kwargs = mock_rag.call_args.kwargs
        assert kwargs["completions_client"] is client
        assert kwargs["retriever"] is retriever
        assert kwargs["prompt_uri"] == "prompts:/custom@v1"
        assert kwargs["n_results"] == 10
        assert kwargs["model"] == "gpt-4o"

    @patch("omnia_support_agents.information.run_rag_pipeline")
    def test_empty_chunks_returns_empty_context(self, mock_rag):
        mock_rag.return_value = RagResult(answer="no info found", retrieved_chunks=[])

        result = _make_agent().handle("obscure question")

        assert result.context == []
        assert result.answer == "no info found"

    @patch("omnia_support_agents.information.run_rag_pipeline")
    def test_passes_optional_rescorers_and_processors(self, mock_rag):
        mock_rag.return_value = RagResult(answer="answer", retrieved_chunks=[])
        rescorers = [MagicMock()]
        processors = [MagicMock()]

        _make_agent(rescorers=rescorers, context_processors=processors).handle("question")

        kwargs = mock_rag.call_args.kwargs
        assert kwargs["rescorers"] is rescorers
        assert kwargs["context_processors"] is processors

    @patch("omnia_support_agents.information.run_rag_pipeline")
    def test_passes_query_interpreter(self, mock_rag):
        mock_rag.return_value = RagResult(answer="answer", retrieved_chunks=[])
        interpreter = MagicMock()

        _make_agent(query_interpreter=interpreter).handle("question")

        assert mock_rag.call_args.kwargs["query_interpreter"] is interpreter

    @patch("omnia_support_agents.information.run_rag_pipeline")
    def test_forwards_use_metadata_filter(self, mock_rag):
        mock_rag.return_value = RagResult(answer="answer", retrieved_chunks=[])

        _make_agent(use_metadata_filter=False).handle("question")

        assert mock_rag.call_args.kwargs["use_metadata_filter"] is False
