"""Tests for the RAG pipeline module."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

from omnia_support_rag.rag import (
    _apply_rescorers,
    _reorder_by_score,
    generate_answer,
    render_messages,
    run_rag_pipeline,
)
from omnia_support_rag.retrievers import (
    BM25Retriever,
    HybridRetriever,
    RetrieverWeight,
    SemanticRetriever,
    _rrf_fuse,
)
from omnia_support_store import KeywordStore, VectorStore, WhereFilter
from omnia_support_store.models import ChunkMetadata, ScoredChunk


def _make_scored_record(
    *,
    document_id: str = "doc-1",
    page_content: str = "Some text",
    chunk_index: int = 0,
    total_chunks: int = 1,
    score: float = 0.1,
) -> ScoredChunk:
    metadata: ChunkMetadata = {
        "document_id": document_id,
        "type": "faq",
        "category": "General",
        "date": "2025-01-01",
        "product_category": "laptops",
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
    }
    return ScoredChunk(
        id=f"{document_id}-chunk-{chunk_index}",
        page_content=page_content,
        metadata=metadata,
        score=score,
    )


def _make_prompt(template: list) -> MagicMock:
    prompt = MagicMock()
    prompt.template = template
    return prompt


def _make_completion(content: str | None = "The answer") -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


class TestSemanticRetriever:
    def test_delegates_to_vector_store(self):
        store = MagicMock(spec=VectorStore)
        store.query.return_value = [_make_scored_record(page_content="text", score=0.2)]
        retriever = SemanticRetriever(store)

        result = retriever.retrieve("query", 3)

        store.query.assert_called_once_with("query", 3, where=None)
        assert len(result) == 1
        assert result[0].page_content == "text"
        assert result[0].score == 0.2

    def test_returns_empty_list_when_no_results(self):
        store = MagicMock(spec=VectorStore)
        store.query.return_value = []
        retriever = SemanticRetriever(store)

        result = retriever.retrieve("query", 5)

        assert result == []

    def test_passes_where_to_store(self):
        store = MagicMock(spec=VectorStore)
        store.query.return_value = []
        retriever = SemanticRetriever(store)
        where: WhereFilter = {"product_category": "laptops"}

        retriever.retrieve("query", 3, where=where)

        store.query.assert_called_once_with("query", 3, where=where)


class TestBM25Retriever:
    def test_delegates_to_keyword_store(self):
        keyword_store = MagicMock(spec=KeywordStore)
        keyword_store.query.return_value = [_make_scored_record(page_content="bm25 text", score=1.5)]
        retriever = BM25Retriever(keyword_store)

        result = retriever.retrieve("query", 3)

        keyword_store.query.assert_called_once_with("query", 3, where=None)
        assert len(result) == 1
        assert result[0].page_content == "bm25 text"
        assert result[0].score == 1.5

    def test_returns_empty_list_when_no_results(self):
        keyword_store = MagicMock(spec=KeywordStore)
        keyword_store.query.return_value = []
        retriever = BM25Retriever(keyword_store)

        result = retriever.retrieve("query", 5)

        assert result == []

    def test_passes_where_to_store(self):
        keyword_store = MagicMock(spec=KeywordStore)
        keyword_store.query.return_value = []
        retriever = BM25Retriever(keyword_store)
        where: WhereFilter = {"product_category": {"$in": ["laptops", "monitors"]}}

        retriever.retrieve("query", 3, where=where)

        keyword_store.query.assert_called_once_with("query", 3, where=where)


class TestHybridRetriever:
    def test_fuses_results_from_both_sources(self):
        r1 = MagicMock()
        r1.retrieve.return_value = [_make_scored_record(document_id="d1", page_content="semantic A", score=0.1)]
        r2 = MagicMock()
        r2.retrieve.return_value = [_make_scored_record(document_id="d2", page_content="keyword B", score=1.5)]
        retriever = HybridRetriever(RetrieverWeight(r1, 1.0), RetrieverWeight(r2, 1.0))

        result = retriever.retrieve("query", 5)

        assert len(result) == 2
        assert {r.metadata["document_id"] for r in result} == {"d1", "d2"}

    def test_overlapping_doc_gets_higher_score(self):
        shared = _make_scored_record(document_id="shared", page_content="both")
        only_first = _make_scored_record(document_id="first-only", page_content="first")
        r1 = MagicMock()
        r1.retrieve.return_value = [shared, only_first]
        r2 = MagicMock()
        r2.retrieve.return_value = [_make_scored_record(document_id="shared", page_content="both", score=1.0)]
        retriever = HybridRetriever(RetrieverWeight(r1, 1.0), RetrieverWeight(r2, 1.0))

        result = retriever.retrieve("query", 5)

        shared_result = next(r for r in result if r.metadata["document_id"] == "shared")
        first_result = next(r for r in result if r.metadata["document_id"] == "first-only")
        assert shared_result.score > first_result.score

    def test_empty_results_from_both(self):
        r1 = MagicMock()
        r1.retrieve.return_value = []
        r2 = MagicMock()
        r2.retrieve.return_value = []
        retriever = HybridRetriever(RetrieverWeight(r1, 1.0), RetrieverWeight(r2, 1.0))

        result = retriever.retrieve("query", 5)

        assert result == []

    def test_one_source_empty(self):
        r1 = MagicMock()
        r1.retrieve.return_value = [_make_scored_record(page_content="only source")]
        r2 = MagicMock()
        r2.retrieve.return_value = []
        retriever = HybridRetriever(RetrieverWeight(r1, 1.0), RetrieverWeight(r2, 1.0))

        result = retriever.retrieve("query", 5)

        assert len(result) == 1
        assert result[0].page_content == "only source"

    def test_custom_weights(self):
        r1 = MagicMock()
        r1.retrieve.return_value = [_make_scored_record(document_id="heavy", page_content="h")]
        r2 = MagicMock()
        r2.retrieve.return_value = [_make_scored_record(document_id="light", page_content="l")]
        retriever = HybridRetriever(RetrieverWeight(r1, 2.0), RetrieverWeight(r2, 1.0))

        result = retriever.retrieve("query", 5)

        heavy = next(r for r in result if r.metadata["document_id"] == "heavy")
        light = next(r for r in result if r.metadata["document_id"] == "light")
        assert heavy.score > light.score

    def test_passes_where_to_child_retrievers(self):
        r1 = MagicMock()
        r1.retrieve.return_value = []
        r2 = MagicMock()
        r2.retrieve.return_value = []
        retriever = HybridRetriever(RetrieverWeight(r1, 1.0), RetrieverWeight(r2, 1.0))
        where: WhereFilter = {"product_category": "laptops"}

        retriever.retrieve("query", 5, where=where)

        r1.retrieve.assert_called_once_with("query", 5, where=where)
        r2.retrieve.assert_called_once_with("query", 5, where=where)


class TestRrfFuse:
    def test_single_ranking(self):
        records = [
            _make_scored_record(document_id="a", page_content="A"),
            _make_scored_record(document_id="b", page_content="B"),
        ]

        fused = _rrf_fuse((1.0, records), k=60, n_results=5)

        assert len(fused) == 2
        assert fused[0].page_content == "A"

    def test_overlapping_docs_score_higher(self):
        ranking1 = [_make_scored_record(document_id="shared", page_content="X")]
        ranking2 = [
            _make_scored_record(document_id="shared", page_content="X"),
            _make_scored_record(document_id="unique", page_content="Y"),
        ]

        fused = _rrf_fuse((1.0, ranking1), (1.0, ranking2), k=60, n_results=5)

        assert fused[0].id == "shared-chunk-0"

    def test_n_results_limits_output(self):
        records = [_make_scored_record(document_id=f"d{i}", page_content=f"chunk {i}") for i in range(10)]

        fused = _rrf_fuse((1.0, records), k=60, n_results=3)

        assert len(fused) == 3

    def test_rank_order_matters(self):
        ranking = [
            _make_scored_record(document_id="first", page_content="1"),
            _make_scored_record(document_id="last", page_content="5"),
        ]

        fused = _rrf_fuse((1.0, ranking), k=60, n_results=5)

        assert fused[0].score > fused[1].score


class _BoostRescorer:
    """Test rescorer that boosts score by customer_satisfaction if present."""

    def rescore(self, chunk: ScoredChunk, intent=None) -> ScoredChunk:  # noqa: ARG002
        sat = chunk.metadata.get("customer_satisfaction", 0.0)
        return replace(chunk, score=chunk.score + sat)


class _MultiplyRescorer:
    def __init__(self, factor: float) -> None:
        self._factor = factor

    def rescore(self, chunk: ScoredChunk, intent=None) -> ScoredChunk:  # noqa: ARG002
        return replace(chunk, score=chunk.score * self._factor)


class _AddRescorer:
    def __init__(self, value: float) -> None:
        self._value = value

    def rescore(self, chunk: ScoredChunk, intent=None) -> ScoredChunk:  # noqa: ARG002
        return replace(chunk, score=chunk.score + self._value)


class _IdentityRescorer:
    def rescore(self, chunk: ScoredChunk, intent=None) -> ScoredChunk:  # noqa: ARG002
        return chunk


class TestApplyRescorers:
    def test_single_rescorer_modifies_score(self):
        records = [
            _make_scored_record(document_id="d1", score=1.0),
            _make_scored_record(document_id="d2", score=2.0),
        ]

        result = _apply_rescorers(records, [_MultiplyRescorer(10)])

        assert result[0].score == 10.0
        assert result[1].score == 20.0

    def test_multiple_rescorers_applied_in_sequence(self):
        records = [_make_scored_record(score=1.0)]

        result = _apply_rescorers(
            records,
            [_AddRescorer(10), _MultiplyRescorer(2)],
        )

        assert result[0].score == 22.0

    def test_does_not_reorder(self):
        records = [
            _make_scored_record(document_id="low", score=1.0),
            _make_scored_record(document_id="high", score=10.0),
        ]

        result = _apply_rescorers(records, [_IdentityRescorer()])

        assert result[0].metadata["document_id"] == "low"
        assert result[1].metadata["document_id"] == "high"


class TestReorderByScore:
    def test_sorts_descending(self):
        records = [
            _make_scored_record(document_id="low", score=1.0),
            _make_scored_record(document_id="high", score=10.0),
        ]

        result = _reorder_by_score(records)

        assert result[0].metadata["document_id"] == "high"
        assert result[1].metadata["document_id"] == "low"

    def test_preserves_order_when_already_sorted(self):
        records = [
            _make_scored_record(document_id="first", score=10.0),
            _make_scored_record(document_id="second", score=5.0),
        ]

        result = _reorder_by_score(records)

        assert result[0].metadata["document_id"] == "first"
        assert result[1].metadata["document_id"] == "second"


class TestRetrieverPreservesOrder:
    def test_semantic_retriever_preserves_store_order(self):
        store = MagicMock(spec=VectorStore)
        store.query.return_value = [
            _make_scored_record(document_id="d1", page_content="A", score=0.1),
            _make_scored_record(document_id="d2", page_content="B", score=0.5),
        ]

        retriever = SemanticRetriever(store)
        result = retriever.retrieve("query", 2)

        assert result[0].metadata["document_id"] == "d1"
        assert result[1].metadata["document_id"] == "d2"

    def test_hybrid_retriever_preserves_rrf_order(self):
        r1 = MagicMock()
        r1.retrieve.return_value = [_make_scored_record(document_id="d1", page_content="first")]
        r2 = MagicMock()
        r2.retrieve.return_value = []

        retriever = HybridRetriever(RetrieverWeight(r1, 1.0), RetrieverWeight(r2, 1.0))
        result = retriever.retrieve("query", 5)

        assert len(result) == 1
        assert result[0].metadata["document_id"] == "d1"


class TestRenderMessages:
    def test_single_variable_substitution(self):
        prompt = _make_prompt([{"role": "user", "content": "Hello {{name}}"}])

        result = render_messages(prompt, name="World")

        assert result == [{"role": "user", "content": "Hello World"}]

    def test_multiple_messages_and_variables(self):
        prompt = _make_prompt(
            [
                {"role": "system", "content": "You are a helper."},
                {"role": "developer", "content": "Context: {{context}}"},
                {"role": "user", "content": "{{question}}"},
            ]
        )

        result = render_messages(prompt, records=[_make_scored_record(page_content="some context")], question="What?")

        assert len(result) == 3
        assert result[0]["content"] == "You are a helper."
        assert result[1]["content"] == "Context: some context"
        assert result[2]["content"] == "What?"

    def test_preserves_role_field(self):
        prompt = _make_prompt(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "usr"},
            ]
        )

        result = render_messages(prompt)

        assert [m["role"] for m in result] == ["system", "user"]

    def test_no_variables_passes_through_unchanged(self):
        prompt = _make_prompt([{"role": "system", "content": "Static text"}])

        result = render_messages(prompt)

        assert result[0]["content"] == "Static text"

    def test_non_dict_entries_filtered(self):
        prompt = _make_prompt([{"role": "user", "content": "ok"}, "not a dict"])

        result = render_messages(prompt)

        assert len(result) == 1
        assert result[0]["content"] == "ok"

    def test_history_spliced_between_preamble_and_current_turn(self):
        prompt = _make_prompt(
            [
                {"role": "system", "content": "System prompt"},
                {"role": "developer", "content": "Context: {{context}}"},
                {"role": "user", "content": "{{question}}"},
            ]
        )
        history = [
            {"role": "developer", "content": "Context: old context"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ]

        result = render_messages(
            prompt,
            history=history,  # ty: ignore[invalid-argument-type]
            records=[_make_scored_record(page_content="new context")],
            question="new question",
        )

        assert len(result) == 6
        assert result[0] == {"role": "system", "content": "System prompt"}
        assert result[1] == {"role": "developer", "content": "Context: old context"}
        assert result[2] == {"role": "user", "content": "old question"}
        assert result[3] == {"role": "assistant", "content": "old answer"}
        assert result[4] == {"role": "developer", "content": "Context: new context"}
        assert result[5] == {"role": "user", "content": "new question"}

    def test_history_filters_out_system_messages(self):
        prompt = _make_prompt(
            [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "{{question}}"},
            ]
        )
        history = [
            {"role": "system", "content": "Should be filtered"},
            {"role": "user", "content": "prior question"},
            {"role": "assistant", "content": "prior answer"},
        ]

        result = render_messages(prompt, history=history, question="new question")  # ty: ignore[invalid-argument-type]

        assert len(result) == 4
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "System prompt"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "user"

    def test_none_history_same_as_no_history(self):
        prompt = _make_prompt(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "{{question}}"},
            ]
        )

        with_none = render_messages(prompt, history=None, question="q")
        without = render_messages(prompt, question="q")

        assert with_none == without


class TestGenerateAnswer:
    def test_returns_response_content(self):
        client = MagicMock()
        client.create.return_value = _make_completion("The answer")
        messages = [{"role": "user", "content": "Hi"}]

        result = generate_answer(client, messages, "gpt-4")  # ty: ignore[invalid-argument-type]

        assert result == "The answer"

    def test_returns_empty_string_when_content_is_none(self):
        client = MagicMock()
        client.create.return_value = _make_completion(None)

        result = generate_answer(client, [], "gpt-4")

        assert result == ""

    def test_passes_correct_parameters(self):
        client = MagicMock()
        client.create.return_value = _make_completion()
        messages = [{"role": "user", "content": "test"}]

        generate_answer(client, messages, "test-model")  # ty: ignore[invalid-argument-type]

        client.create.assert_called_once_with(
            model="test-model",
            messages=messages,
            temperature=0,
        )


@patch("omnia_support_rag.rag.mlflow.genai.load_prompt")
class TestRunRagPipeline:
    def test_orchestrates_retrieve_render_generate(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(
            [
                {"role": "user", "content": "{{context}} {{question}}"},
            ]
        )
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            _make_scored_record(page_content="chunk A"),
            _make_scored_record(page_content="chunk B", document_id="doc-2"),
        ]
        client = MagicMock()
        client.create.return_value = _make_completion("answer text")

        result = run_rag_pipeline(
            "my question",
            completions_client=client,
            retriever=retriever,
            prompt_uri="prompts:/test@prod",
            n_results=5,
            model="gpt-4",
        )

        assert result.answer == "answer text"
        assert len(result.retrieved_chunks) == 2

    def test_calls_load_prompt_with_uri(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(
            [
                {"role": "user", "content": "{{context}} {{question}}"},
            ]
        )
        retriever = MagicMock()
        retriever.retrieve.return_value = [_make_scored_record(page_content="chunk")]
        client = MagicMock()
        client.create.return_value = _make_completion()

        run_rag_pipeline(
            "q",
            completions_client=client,
            retriever=retriever,
            prompt_uri="prompts:/my_prompt@prod",
            n_results=1,
            model="gpt-4",
        )

        mock_load_prompt.assert_called_once_with("prompts:/my_prompt@prod")

    def test_passes_history_to_render_messages(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(
            [
                {"role": "system", "content": "System"},
                {"role": "developer", "content": "{{context}}"},
                {"role": "user", "content": "{{question}}"},
            ]
        )
        retriever = MagicMock()
        retriever.retrieve.return_value = [_make_scored_record(page_content="new chunk")]
        client = MagicMock()
        client.create.return_value = _make_completion("answer")
        history = [
            {"role": "developer", "content": "old context"},
            {"role": "user", "content": "old q"},
            {"role": "assistant", "content": "old a"},
        ]

        run_rag_pipeline(
            "new q",
            completions_client=client,
            retriever=retriever,
            prompt_uri="prompts:/p@prod",
            n_results=1,
            model="gpt-4",
            history=history,  # ty: ignore[invalid-argument-type]
        )

        messages = client.create.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "System"}
        assert messages[1] == {"role": "developer", "content": "old context"}
        assert messages[2] == {"role": "user", "content": "old q"}
        assert messages[3] == {"role": "assistant", "content": "old a"}
        assert messages[4] == {"role": "developer", "content": "new chunk"}
        assert messages[5] == {"role": "user", "content": "new q"}

    def test_uses_query_interpreter_when_provided(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(
            [
                {"role": "user", "content": "{{context}} {{question}}"},
            ]
        )
        retriever = MagicMock()
        retriever.retrieve.return_value = [_make_scored_record(page_content="chunk")]
        client = MagicMock()
        client.create.return_value = _make_completion("answer")
        interpreter = MagicMock()
        interpreter.interpret.return_value = MagicMock(
            query="standalone question",
            effective_retrieval_query="standalone question",
            product_categories=["laptops"],
            doc_type_relevance=MagicMock(policy=5, faq=8, support_ticket=3),
        )
        history = [
            {"role": "user", "content": "prior"},
            {"role": "assistant", "content": "reply"},
        ]

        run_rag_pipeline(
            "follow-up",
            completions_client=client,
            retriever=retriever,
            prompt_uri="prompts:/p@prod",
            n_results=1,
            model="gpt-4",
            history=history,  # ty: ignore[invalid-argument-type]
            query_interpreter=interpreter,
        )

        interpreter.interpret.assert_called_once_with("follow-up", history)
        where_arg = retriever.retrieve.call_args.kwargs["where"]
        assert set(where_arg["product_category"]["$in"]) == {"laptops", "all", "none"}

    def test_passthrough_interpreter_preserves_query_with_no_filter(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(
            [
                {"role": "user", "content": "{{context}} {{question}}"},
            ]
        )
        retriever = MagicMock()
        retriever.retrieve.return_value = [_make_scored_record(page_content="chunk")]
        client = MagicMock()
        client.create.return_value = _make_completion("answer")

        run_rag_pipeline(
            "my question",
            completions_client=client,
            retriever=retriever,
            prompt_uri="prompts:/p@prod",
            n_results=1,
            model="gpt-4",
        )

        retriever.retrieve.assert_called_once_with("my question", 1, where=None)

    def test_interpreter_multiple_categories_builds_in_filter(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(
            [
                {"role": "user", "content": "{{context}} {{question}}"},
            ]
        )
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        client = MagicMock()
        client.create.return_value = _make_completion("answer")
        interpreter = MagicMock()
        interpreter.interpret.return_value = MagicMock(
            query="question",
            effective_retrieval_query="question",
            product_categories=["laptops", "monitors"],
            doc_type_relevance=MagicMock(policy=5, faq=5, support_ticket=5),
        )

        run_rag_pipeline(
            "question",
            completions_client=client,
            retriever=retriever,
            prompt_uri="prompts:/p@prod",
            n_results=1,
            model="gpt-4",
            query_interpreter=interpreter,
        )

        where_arg = retriever.retrieve.call_args.kwargs["where"]
        assert set(where_arg["product_category"]["$in"]) == {"laptops", "monitors", "all", "none"}

    def test_interpreter_empty_categories_no_filter(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(
            [
                {"role": "user", "content": "{{context}} {{question}}"},
            ]
        )
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        client = MagicMock()
        client.create.return_value = _make_completion("answer")
        interpreter = MagicMock()
        interpreter.interpret.return_value = MagicMock(
            query="general question",
            effective_retrieval_query="general question",
            product_categories=[],
            doc_type_relevance=MagicMock(policy=5, faq=5, support_ticket=5),
        )

        run_rag_pipeline(
            "general question",
            completions_client=client,
            retriever=retriever,
            prompt_uri="prompts:/p@prod",
            n_results=1,
            model="gpt-4",
            query_interpreter=interpreter,
        )

        retriever.retrieve.assert_called_once_with("general question", 1, where=None)

    def test_context_joined_with_separator(self, mock_load_prompt):
        mock_load_prompt.return_value = _make_prompt(
            [
                {"role": "user", "content": "{{context}}"},
            ]
        )
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            _make_scored_record(page_content="first"),
            _make_scored_record(page_content="second", document_id="doc-2"),
            _make_scored_record(page_content="third", document_id="doc-3"),
        ]
        client = MagicMock()
        client.create.return_value = _make_completion()

        result = run_rag_pipeline(
            "q",
            completions_client=client,
            retriever=retriever,
            prompt_uri="prompts:/p@prod",
            n_results=3,
            model="gpt-4",
        )

        assert [r.page_content for r in result.retrieved_chunks] == ["first", "second", "third"]
